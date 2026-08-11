"""Train deterministic LSTM forecasting baselines for Task 08A."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from va_diff.data.timeseries import MODEL_FEATURES

SEED = 42
CONTEXT_LENGTH = 168
HORIZONS = (1, 6, 24)
HIDDEN_SIZE = 16
NUM_LAYERS = 1
BATCH_SIZE = 1024
EPOCHS = 4
PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


class LSTMForecaster(nn.Module):
    """One-layer sequence-to-vector LSTM; output length equals forecast horizon."""

    def __init__(self, input_size: int = 4, hidden_size: int = HIDDEN_SIZE, horizon: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1])


def load_scaled_features(features_path: Path, scaler_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    features = pd.read_parquet(features_path)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler_info = scaler["scaler"]
    if scaler_info["targets_scaled"] is not False:
        raise ValueError("Task 07A scaler artifact does not guarantee unscaled targets")
    if tuple(scaler_info["features"]) != MODEL_FEATURES:
        raise ValueError("Task 07A scaler feature order does not match Task 08A")
    if "rv_168h" in MODEL_FEATURES:
        raise ValueError("diagnostic rv_168h must not be used by Task 08A")
    scaled = features.copy()
    for feature in MODEL_FEATURES:
        values = scaled[feature].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"infinite feature value found in {feature}")
        scaled[feature] = (values - float(scaler_info["mean"][feature])) / float(
            scaler_info["scale"][feature]
        )
    return scaled, scaler


def build_window_dataset(
    scaled_features: pd.DataFrame,
    original_returns: np.ndarray,
    sample_index: pd.DataFrame,
    *,
    split: str,
    horizon: int,
) -> TensorDataset:
    """Materialize only authoritative, already-validated Task 06 windows."""
    rows = sample_index.loc[
        (sample_index["split"] == split) & (sample_index["horizon"] == horizon)
    ].reset_index(drop=True)
    endpoints = rows["endpoint_row"].to_numpy(dtype=np.int64)
    offsets = np.arange(CONTEXT_LENGTH - 1, -1, -1, dtype=np.int64)
    feature_values = scaled_features[list(MODEL_FEATURES)].to_numpy(dtype=np.float32)
    contexts = feature_values[endpoints[:, None] - offsets[None, :]]
    targets = np.stack(
        [original_returns[endpoints + step] for step in range(1, horizon + 1)], axis=1
    ).astype(np.float32)
    if not np.isfinite(contexts).all() or not np.isfinite(targets).all():
        raise ValueError(f"non-finite Task 06 window for {split}, horizon {horizon}")
    return TensorDataset(torch.from_numpy(contexts), torch.from_numpy(targets))


def train_one_horizon(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    horizon: int,
) -> tuple[LSTMForecaster, dict[str, object]]:
    seed_everything(SEED + horizon)
    model = LSTMForecaster(horizon=horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at horizon {horizon}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                loss = loss_fn(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite validation loss at horizon {horizon}, epoch {epoch}")
                validation_losses.append(float(loss))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def predict(model: LSTMForecaster, dataset: TensorDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _targets in loader:
            output = model(inputs).numpy()
            if not np.isfinite(output).all():
                raise ValueError("non-finite LSTM prediction")
            values.append(output)
    return np.concatenate(values, axis=0) if values else np.empty((0, dataset.horizon))


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, horizon), group in predictions.groupby(
        ["model", "split", "horizon"], sort=False
    ):
        error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        rows.append(
            {
                "model": model,
                "split": split,
                "horizon": horizon,
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    return pd.DataFrame(rows, columns=["model", "split", "horizon", "n", "mae", "rmse"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--task06-manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--regime-index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs/predictions/task08a_lstm_predictions.parquet"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/task08a_metrics.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata/task08a_model_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/metadata/task08a_validation_report.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    args = parser.parse_args()
    seed_everything()

    features, _scaler_artifact = load_scaled_features(args.features_path, args.scaler_path)
    sample_index = pd.read_parquet(args.sample_index_path)
    manifest = json.loads(args.task06_manifest_path.read_text(encoding="utf-8"))
    sample_index_checksum = sha256(args.sample_index_path)
    if manifest["sample_index_sha256"] != sample_index_checksum:
        raise ValueError("Task 06 sample index checksum does not match its manifest")
    sample_index["endpoint_timestamp"] = pd.to_datetime(sample_index["endpoint_timestamp"], utc=True)
    sample_index["target_end_timestamp"] = pd.to_datetime(sample_index["target_end_timestamp"], utc=True)
    regime_index = pd.read_parquet(args.regime_index_path)
    regime_index["timestamp"] = pd.to_datetime(regime_index["timestamp"], utc=True)
    regime_by_timestamp = regime_index.set_index("timestamp")["regime"]
    original_features = pd.read_parquet(args.features_path)
    original_features["timestamp"] = pd.to_datetime(original_features["timestamp"], utc=True)
    if not features["timestamp"].equals(original_features["timestamp"]):
        raise ValueError("scaled and original feature timestamps differ")
    original_returns = original_features["log_return"].to_numpy(dtype=float)
    all_frames: list[pd.DataFrame] = []
    training_metadata: dict[str, object] = {}
    for horizon in HORIZONS:
        train_dataset = build_window_dataset(
            features, original_returns, sample_index, split="train", horizon=horizon
        )
        validation_dataset = build_window_dataset(
            features, original_returns, sample_index, split="validation", horizon=horizon
        )
        test_dataset = build_window_dataset(
            features, original_returns, sample_index, split="test", horizon=horizon
        )
        model, fit_metadata = train_one_horizon(train_dataset, validation_dataset, horizon)
        training_metadata[str(horizon)] = fit_metadata
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": "08A",
                "seed": SEED,
                "horizon": horizon,
                "architecture": {"input_size": 4, "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS},
                "state_dict": model.state_dict(),
                "fit": fit_metadata,
            },
            args.checkpoint_dir / f"task08a_lstm_h{horizon}.pt",
        )
        for split, dataset in (("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset)):
            predictions = predict(model, dataset)
            rows = sample_index.loc[
                (sample_index["split"] == split) & (sample_index["horizon"] == horizon)
            ].reset_index(drop=True)
            y_true = original_returns[rows["endpoint_row"].to_numpy(dtype=int) + horizon]
            frame = pd.DataFrame(
                {
                    "timestamp": rows["target_end_timestamp"],
                    "origin_timestamp": rows["endpoint_timestamp"],
                    "split": split,
                    "horizon": horizon,
                    "model": "lstm",
                    "y_true": y_true,
                    "y_pred": predictions[:, -1],
                    "regime": regime_by_timestamp.reindex(rows["endpoint_timestamp"]).to_numpy(),
                    "config_model_metadata_ref": "outputs/metadata/task08a_model_metadata.json",
                }
            )
            all_frames.append(frame)
    predictions = pd.concat(all_frames, ignore_index=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions["origin_timestamp"] = pd.to_datetime(predictions["origin_timestamp"], utc=True)
    predictions = predictions[
        ["timestamp", "origin_timestamp", "split", "horizon", "model", "y_true", "y_pred", "regime", "config_model_metadata_ref"]
    ]
    metrics = metric_rows(predictions)
    zero_metrics_path = Path("outputs/metrics/task07b_metrics.csv")
    zero_metrics = pd.read_csv(zero_metrics_path) if zero_metrics_path.exists() else pd.DataFrame()
    test_comparison = metrics.loc[metrics["split"] == "test"].merge(
        zero_metrics.loc[
            (zero_metrics["model"] == "zero_return") & (zero_metrics["split"] == "test"),
            ["horizon", "mae", "rmse"],
        ],
        on="horizon",
        suffixes=("_lstm", "_zero_return"),
    )
    args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.prediction_path, index=False, engine="pyarrow")
    metrics.to_csv(args.metrics_path, index=False)
    saved_predictions = pd.read_parquet(args.prediction_path)
    saved_metrics = metric_rows(saved_predictions)
    metrics_recomputed = bool(
        len(saved_metrics) == len(metrics)
        and np.allclose(saved_metrics["mae"], metrics["mae"])
        and np.allclose(saved_metrics["rmse"], metrics["rmse"])
        and np.array_equal(saved_metrics["n"], metrics["n"])
    )
    metadata = {
        "task": "08A",
        "seed": SEED,
        "architecture": f"LSTM(input_size=4, hidden_size={HIDDEN_SIZE}, num_layers=1, batch_first=True) -> Linear({HIDDEN_SIZE}, horizon)",
        "features": list(MODEL_FEATURES),
        "target": "original unscaled log_return",
        "training": {"batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "selection": "validation early stopping only"},
        "horizon_fit": training_metadata,
        "sample_index_sha256": sample_index_checksum,
        "scaler_sha256": sha256(args.scaler_path),
        "scaler_reference": str(args.scaler_path),
        "test_used_for_tuning": False,
        "prediction_schema": list(predictions.columns),
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    expected_rows = len(sample_index)
    timestamps_match = True
    for split in sample_index["split"].unique():
        for horizon in sample_index["horizon"].unique():
            expected_timestamps = sample_index.loc[
                (sample_index["split"] == split) & (sample_index["horizon"] == horizon),
                "target_end_timestamp",
            ].sort_values().reset_index(drop=True)
            actual_timestamps = saved_predictions.loc[
                (saved_predictions["split"] == split) & (saved_predictions["horizon"] == horizon),
                "timestamp",
            ].sort_values().reset_index(drop=True)
            timestamps_match &= bool(expected_timestamps.equals(actual_timestamps))
    validation = {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "expected_prediction_rows": expected_rows,
        "sample_counts": {f"{row.split}/horizon_{row.horizon}": int(row.n) for row in metrics.itertuples()},
        "sample_index_sha256": sample_index_checksum,
        "timestamps_match_authoritative_index": bool(
            len(predictions) == expected_rows and predictions["timestamp"].notna().all() and timestamps_match
        ),
        "first_validation_target": str(predictions.loc[predictions["split"] == "validation", "timestamp"].min()),
        "first_test_target": str(predictions.loc[predictions["split"] == "test", "timestamp"].min()),
        "targets_unscaled": True,
        "test_used_for_tuning": False,
        "no_nan_inf_predictions": bool(np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all()),
        "metrics_recomputed_from_saved_predictions": metrics_recomputed,
        "test_comparison_vs_zero_return": test_comparison.to_dict(orient="records"),
    }
    validation["status"] = "PASS" if all(
        [
            validation["timestamps_match_authoritative_index"],
            validation["prediction_rows"] == validation["expected_prediction_rows"],
            validation["targets_unscaled"],
            validation["test_used_for_tuning"] is False,
            validation["no_nan_inf_predictions"],
            validation["metrics_recomputed_from_saved_predictions"],
        ]
    ) else "FAIL"
    args.report_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "metrics": metrics.to_dict(orient="records"), "validation": validation}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
