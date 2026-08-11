"""Train a deterministic lightweight Transformer forecasting baseline for Task 08B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.fit_task08a_lstm import (
        BATCH_SIZE,
        CONTEXT_LENGTH,
        EPOCHS,
        HORIZONS,
        LEARNING_RATE,
        PATIENCE,
        WEIGHT_DECAY,
        build_window_dataset,
        load_scaled_features,
        metric_rows,
        seed_everything,
        sha256,
    )
except ModuleNotFoundError:
    from fit_task08a_lstm import (
        BATCH_SIZE,
        CONTEXT_LENGTH,
        EPOCHS,
        HORIZONS,
        LEARNING_RATE,
        PATIENCE,
        WEIGHT_DECAY,
        build_window_dataset,
        load_scaled_features,
        metric_rows,
        seed_everything,
        sha256,
    )

SEED = 42
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2
FEEDFORWARD_DIM = 64
DROPOUT = 0.1
FRONT_END_KERNEL = 8
FRONT_END_STRIDE = 8
FRONT_END_PADDING = 4
TOKEN_LENGTH = (CONTEXT_LENGTH + 2 * FRONT_END_PADDING - FRONT_END_KERNEL) // FRONT_END_STRIDE + 1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = TOKEN_LENGTH) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.size(1)]


class TransformerForecaster(nn.Module):
    """Transformer encoder using the final context token for horizon forecasts."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.front_end = nn.Conv1d(
            4,
            D_MODEL,
            kernel_size=FRONT_END_KERNEL,
            stride=FRONT_END_STRIDE,
            padding=FRONT_END_PADDING,
        )
        self.position = SinusoidalPositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.head = nn.Linear(D_MODEL, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The convolution scans all 168 observations before tokenization.
        tokens = self.front_end(inputs.transpose(1, 2)).transpose(1, 2)
        encoded = self.encoder(self.position(tokens))
        return self.head(encoded[:, -1])


def train_one_horizon(
    train_dataset: TensorDataset, validation_dataset: TensorDataset, horizon: int
) -> tuple[TransformerForecaster, dict[str, object]]:
    seed_everything(SEED + horizon)
    model = TransformerForecaster(horizon)
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
            loss = loss_fn(model(inputs), targets)
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
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}


def predict(model: TransformerForecaster, dataset: TensorDataset) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _targets in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            output = model(inputs).numpy()
            if not np.isfinite(output).all():
                raise ValueError("non-finite Transformer prediction")
            values.append(output)
    return np.concatenate(values, axis=0) if values else np.empty((0, dataset.tensors[1].shape[1]))


def all_context_observations_reach_front_end() -> bool:
    coverage = np.zeros(CONTEXT_LENGTH, dtype=bool)
    for output_index in range(TOKEN_LENGTH):
        start = output_index * FRONT_END_STRIDE - FRONT_END_PADDING
        stop = start + FRONT_END_KERNEL
        coverage[max(0, start) : min(CONTEXT_LENGTH, stop)] = True
    return bool(coverage.all())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--task06-manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--regime-index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs/predictions/task08b_transformer_predictions.parquet"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/task08b_metrics.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata/task08b_model_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/metadata/task08b_validation_report.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    args = parser.parse_args()
    seed_everything(SEED)
    features, _scaler_artifact = load_scaled_features(args.features_path, args.scaler_path)
    original_features = pd.read_parquet(args.features_path)
    original_features["timestamp"] = pd.to_datetime(original_features["timestamp"], utc=True)
    original_returns = original_features["log_return"].to_numpy(dtype=float)
    sample_index = pd.read_parquet(args.sample_index_path)
    sample_index["endpoint_timestamp"] = pd.to_datetime(sample_index["endpoint_timestamp"], utc=True)
    sample_index["target_end_timestamp"] = pd.to_datetime(sample_index["target_end_timestamp"], utc=True)
    manifest = json.loads(args.task06_manifest_path.read_text(encoding="utf-8"))
    sample_index_checksum = sha256(args.sample_index_path)
    if manifest["sample_index_sha256"] != sample_index_checksum:
        raise ValueError("Task 06 sample index checksum does not match its manifest")
    regime_index = pd.read_parquet(args.regime_index_path)
    regime_index["timestamp"] = pd.to_datetime(regime_index["timestamp"], utc=True)
    regime_by_timestamp = regime_index.set_index("timestamp")["regime"]
    all_frames: list[pd.DataFrame] = []
    horizon_metadata: dict[str, object] = {}
    for horizon in HORIZONS:
        datasets = {
            split: build_window_dataset(
                features, original_returns, sample_index, split=split, horizon=horizon
            )
            for split in ("train", "validation", "test")
        }
        model, fit_metadata = train_one_horizon(datasets["train"], datasets["validation"], horizon)
        horizon_metadata[str(horizon)] = fit_metadata
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": "08B",
                "seed": SEED,
                "horizon": horizon,
                "architecture": {
                    "input_size": 4,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "feedforward_dim": FEEDFORWARD_DIM,
                    "dropout": DROPOUT,
                    "positional_encoding": "sinusoidal",
                    "front_end": {
                        "type": "learnable_conv1d",
                        "kernel_size": FRONT_END_KERNEL,
                        "stride": FRONT_END_STRIDE,
                        "padding": FRONT_END_PADDING,
                        "uses_all_context_observations": True,
                    },
                    "token_length": TOKEN_LENGTH,
                },
                "state_dict": model.state_dict(),
                "fit": fit_metadata,
            },
            args.checkpoint_dir / f"task08b_transformer_h{horizon}.pt",
        )
        for split in ("train", "validation", "test"):
            predictions = predict(model, datasets[split])
            rows = sample_index.loc[
                (sample_index["split"] == split) & (sample_index["horizon"] == horizon)
            ].reset_index(drop=True)
            frame = pd.DataFrame(
                {
                    "timestamp": rows["target_end_timestamp"],
                    "origin_timestamp": rows["endpoint_timestamp"],
                    "split": split,
                    "horizon": horizon,
                    "model": "transformer",
                    "y_true": original_returns[rows["endpoint_row"].to_numpy(dtype=int) + horizon],
                    "y_pred": predictions[:, -1],
                    "regime": regime_by_timestamp.reindex(rows["endpoint_timestamp"]).to_numpy(),
                    "config_model_metadata_ref": "outputs/metadata/task08b_model_metadata.json",
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
    comparison = metrics.loc[metrics["split"] == "test"].copy()
    reference_frames = []
    for path, model_name in (
        (Path("outputs/metrics/task07b_metrics.csv"), "zero_return"),
        (Path("outputs/metrics/task08a_metrics.csv"), "lstm"),
    ):
        reference = pd.read_csv(path)
        reference_frames.append(reference.loc[reference["model"] == model_name, ["split", "horizon", "mae", "rmse"]].rename(columns={"mae": f"mae_{model_name}", "rmse": f"rmse_{model_name}"}))
    for reference in reference_frames:
        comparison = comparison.merge(reference.loc[reference["split"] == "test"], on=["split", "horizon"], how="left")
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
        "task": "08B",
        "seed": SEED,
        "architecture": "Conv1d(4 -> 32, kernel=8, stride=4, padding=4; all 168 inputs) -> TransformerEncoder(d_model=32, nhead=4, layers=2, feedforward=64, dropout=0.1, sinusoidal positional encoding) -> Linear(32, horizon)",
        "features": ["log_return", "log_volume", "high_low_range", "rv_24h"],
        "target": "original unscaled log_return",
        "training": {"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": EPOCHS, "selection": "validation-only model selection"},
        "horizon_fit": horizon_metadata,
        "sample_index_sha256": sample_index_checksum,
        "scaler_sha256": sha256(args.scaler_path),
        "test_used_for_tuning": False,
        "prediction_schema": list(predictions.columns),
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    timestamps_match = True
    for split in sample_index["split"].unique():
        for horizon in sample_index["horizon"].unique():
            expected = sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon), "target_end_timestamp"].sort_values().reset_index(drop=True)
            actual = saved_predictions.loc[(saved_predictions["split"] == split) & (saved_predictions["horizon"] == horizon), "timestamp"].sort_values().reset_index(drop=True)
            timestamps_match &= bool(expected.equals(actual))
    validation = {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "expected_prediction_rows": len(sample_index),
        "sample_counts": {f"{row.split}/horizon_{row.horizon}": int(row.n) for row in metrics.itertuples()},
        "sample_index_sha256": sample_index_checksum,
        "timestamps_match_authoritative_index": bool(timestamps_match),
        "first_validation_target": str(predictions.loc[predictions["split"] == "validation", "timestamp"].min()),
        "first_test_target": str(predictions.loc[predictions["split"] == "test", "timestamp"].min()),
        "targets_unscaled": True,
        "test_used_for_tuning": False,
        "no_nan_inf_predictions": bool(np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all()),
        "all_168_context_observations_contribute": all_context_observations_reach_front_end(),
        "finite_training_validation_loss": bool(
            all(
                np.isfinite(record["train_loss"]) and np.isfinite(record["validation_loss"])
                for fit in horizon_metadata.values()
                for record in fit["history"]
            )
        ),
        "metrics_recomputed_from_saved_predictions": metrics_recomputed,
        "test_comparison": comparison.to_dict(orient="records"),
    }
    validation["status"] = "PASS" if all(
        [validation["timestamps_match_authoritative_index"], validation["prediction_rows"] == validation["expected_prediction_rows"], validation["targets_unscaled"], validation["test_used_for_tuning"] is False, validation["no_nan_inf_predictions"], validation["finite_training_validation_loss"], validation["all_168_context_observations_contribute"], validation["metrics_recomputed_from_saved_predictions"]]
    ) else "FAIL"
    args.report_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "metrics": metrics.to_dict(orient="records"), "validation": validation}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
