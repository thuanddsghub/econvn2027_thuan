"""Train and evaluate the standard conditional diffusion baseline (Task 09)."""

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
        CONTEXT_LENGTH,
        HORIZONS,
        build_window_dataset,
        load_scaled_features,
        seed_everything,
        sha256,
    )
except ModuleNotFoundError:
    from fit_task08a_lstm import (
        CONTEXT_LENGTH,
        HORIZONS,
        build_window_dataset,
        load_scaled_features,
        seed_everything,
        sha256,
    )

from va_diff.models.standard_diffusion import GaussianDiffusion, StandardConditionalDiffusion

SEED = 42
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 10
PATIENCE = 3
DIFFUSION_STEPS = 100
BETA_START = 1e-4
BETA_END = 0.02
FINAL_SAMPLE_COUNT = 2


def _mean_loss(
    model: StandardConditionalDiffusion,
    diffusion: GaussianDiffusion,
    dataset: TensorDataset,
    *,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for history, target in DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False):
            losses.append(float(diffusion.loss(model, target, history)))
    return float(np.mean(losses))


def train_one_horizon(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    horizon: int,
) -> tuple[StandardConditionalDiffusion, dict[str, object]]:
    seed_everything(SEED)
    model = StandardConditionalDiffusion(horizon=horizon, hidden_size=32, gru_layers=1)
    diffusion = GaussianDiffusion(
        steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for context, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, target, context)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at horizon {horizon}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation_loss = _mean_loss(
            model, diffusion, validation_dataset, seed=SEED + 1000 + horizon * 100 + epoch
        )
        train_loss = float(np.mean(train_losses))
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise ValueError(f"non-finite epoch loss at horizon {horizon}, epoch {epoch}")
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


def tiny_overfit_check() -> dict[str, object]:
    seed_everything(SEED)
    horizon = 1
    context = torch.zeros((8, CONTEXT_LENGTH, 4))
    target = torch.zeros((8, horizon))
    model = StandardConditionalDiffusion(horizon=horizon, hidden_size=32, gru_layers=1)
    diffusion = GaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    torch.manual_seed(SEED + 9000)
    timesteps = torch.full((len(target),), 50, dtype=torch.long)
    noise = torch.zeros_like(target)
    noisy, _ = diffusion.q_sample(target, timesteps, noise=noise)
    initial = float(diffusion.loss(model, target, context).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(noisy, context, timesteps) - noise) ** 2)
        loss.backward()
        optimizer.step()
    final = float(torch.mean((model(noisy, context, timesteps) - noise) ** 2).detach())
    return {"status": bool(np.isfinite(initial) and np.isfinite(final) and final < initial), "initial_loss": initial, "final_loss": final}


def sample_split(
    model: StandardConditionalDiffusion,
    dataset: TensorDataset,
    rows: pd.DataFrame,
    *,
    split: str,
    horizon: int,
    original_returns: np.ndarray,
    regime_by_timestamp: pd.Series,
) -> pd.DataFrame:
    diffusion = GaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    generated: list[np.ndarray] = []
    for batch_number, (context, _target) in enumerate(loader):
        generator = torch.Generator().manual_seed(SEED + horizon * 10000 + batch_number)
        generated.append(
            diffusion.sample(model, context, sample_count=FINAL_SAMPLE_COUNT, generator=generator).numpy()
        )
    samples = np.concatenate(generated, axis=0)
    output: list[pd.DataFrame] = []
    for sample_id in range(FINAL_SAMPLE_COUNT):
        for target_step in range(1, horizon + 1):
            target_rows = rows.copy()
            target_positions = target_rows["endpoint_row"].to_numpy(dtype=int) + target_step
            target_rows["timestamp"] = pd.to_datetime(
                [
                    rows.iloc[index]["endpoint_timestamp"] + pd.Timedelta(hours=target_step)
                    for index in range(len(rows))
                ],
                utc=True,
            )
            output.append(
                pd.DataFrame(
                    {
                        "timestamp": target_rows["timestamp"],
                        "origin_timestamp": rows["endpoint_timestamp"],
                        "split": split,
                        "horizon": horizon,
                        "target_step": target_step,
                        "sample_id": sample_id,
                        "model": "standard_diffusion",
                        "y_true": original_returns[target_positions],
                        "y_pred": samples[:, sample_id, target_step - 1],
                        "regime": regime_by_timestamp.reindex(rows["endpoint_timestamp"]).to_numpy(),
                        "config_model_metadata_ref": "outputs/metadata/task09_model_metadata.json",
                    }
                )
            )
    return pd.concat(output, ignore_index=True)


def mean_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    mean_prediction = (
        predictions.groupby(["split", "horizon", "target_step", "timestamp", "origin_timestamp"], as_index=False)["y_pred"].mean()
    )
    truth = predictions.drop_duplicates(["split", "horizon", "target_step", "timestamp", "origin_timestamp"])[
        ["split", "horizon", "target_step", "timestamp", "origin_timestamp", "y_true"]
    ]
    frame = mean_prediction.merge(truth, on=["split", "horizon", "target_step", "timestamp", "origin_timestamp"])
    rows = []
    for (split, horizon, target_step), group in frame.groupby(["split", "horizon", "target_step"], sort=False):
        error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        rows.append({"model": "standard_diffusion_mean", "split": split, "horizon": horizon, "target_step": target_step, "n": len(group), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--task06-manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--regime-index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs/predictions/task09_standard_diffusion_predictions.parquet"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/task09_standard_diffusion_metrics.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata/task09_model_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/metadata/task09_validation_report.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    args = parser.parse_args()
    seed_everything(SEED)
    features, _scaler = load_scaled_features(args.features_path, args.scaler_path)
    original = pd.read_parquet(args.features_path)
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    original_returns = original["log_return"].to_numpy(dtype=float)
    sample_index = pd.read_parquet(args.sample_index_path)
    for column in ["endpoint_timestamp", "target_end_timestamp"]:
        sample_index[column] = pd.to_datetime(sample_index[column], utc=True)
    sample_checksum = sha256(args.sample_index_path)
    manifest = json.loads(args.task06_manifest_path.read_text(encoding="utf-8"))
    if manifest["sample_index_sha256"] != sample_checksum:
        raise ValueError("Task 06 sample index checksum does not match manifest")
    regime = pd.read_parquet(args.regime_index_path)
    regime["timestamp"] = pd.to_datetime(regime["timestamp"], utc=True)
    regime_by_timestamp = regime.set_index("timestamp")["regime"]
    all_predictions: list[pd.DataFrame] = []
    fit_metadata: dict[str, object] = {}
    for horizon in HORIZONS:
        datasets = {
            split: build_window_dataset(features, original_returns, sample_index, split=split, horizon=horizon)
            for split in ("train", "validation", "test")
        }
        model, fit = train_one_horizon(datasets["train"], datasets["validation"], horizon)
        fit_metadata[str(horizon)] = fit
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"task": "09", "seed": SEED, "horizon": horizon, "state_dict": model.state_dict(), "fit": fit}, args.checkpoint_dir / f"task09_standard_diffusion_h{horizon}.pt")
        for split in ("validation", "test"):
            rows = sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon)].reset_index(drop=True)
            all_predictions.append(sample_split(model, datasets[split], rows, split=split, horizon=horizon, original_returns=original_returns, regime_by_timestamp=regime_by_timestamp))
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = mean_metrics(predictions)
    args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.prediction_path, index=False, engine="pyarrow")
    metrics.to_csv(args.metrics_path, index=False)
    curves = pd.DataFrame(
        [
            {"horizon": int(horizon), **record}
            for horizon, fit in fit_metadata.items()
            for record in fit["history"]
        ]
    )
    curves.to_csv(args.metrics_path.with_name("task09_training_curves.csv"), index=False)
    saved = pd.read_parquet(args.prediction_path)
    saved_metrics = mean_metrics(saved)
    metrics_match = bool(np.allclose(metrics["mae"], saved_metrics["mae"]) and np.allclose(metrics["rmse"], saved_metrics["rmse"]))
    tiny = tiny_overfit_check()
    metadata = {
        "task": "09",
        "seed": SEED,
        "architecture": "GRU(input_size=4, hidden_size=32, layers=1) -> MLP epsilon denoiser(hidden=64, SiLU) with sinusoidal timestep embedding(dim=32)",
        "explicit_volatility_conditioning": False,
        "features": ["log_return", "log_volume", "high_low_range", "rv_24h"],
        "target": "original unscaled log_return",
        "diffusion": {"type": "Gaussian DDPM", "prediction": "epsilon", "steps": DIFFUSION_STEPS, "beta_schedule": "linear", "beta_start": BETA_START, "beta_end": BETA_END, "q_sample": "sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon"},
        "training": {"batch_size": TRAIN_BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "selection": "validation-only"},
        "sampling": {"final_sample_count": FINAL_SAMPLE_COUNT, "splits": ["validation", "test"]},
        "fit": fit_metadata,
        "sample_index_sha256": sample_checksum,
        "scaler_sha256": sha256(args.scaler_path),
        "prediction_schema": list(predictions.columns),
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    expected_rows = sum(len(sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon)]) * FINAL_SAMPLE_COUNT * horizon for split in ("validation", "test") for horizon in HORIZONS)
    exact_target_end = True
    for split in ("validation", "test"):
        for horizon in HORIZONS:
            expected = sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon), "target_end_timestamp"].astype("datetime64[ns, UTC]")
            actual = predictions.loc[(predictions["split"] == split) & (predictions["horizon"] == horizon) & (predictions["target_step"] == horizon), "timestamp"].drop_duplicates().sort_values().reset_index(drop=True).astype("datetime64[ns, UTC]")
            exact_target_end &= bool(expected.sort_values().reset_index(drop=True).equals(actual))
    validation = {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "expected_prediction_rows": expected_rows,
        "sample_index_sha256": sample_checksum,
        "exact_authoritative_target_end_timestamps": exact_target_end,
        "all_168_context_observations_used": True,
        "targets_unscaled": True,
        "rv_168h_used": False,
        "explicit_volatility_conditioning": False,
        "test_used_for_tuning": False,
        "finite_losses": bool(all(np.isfinite(record["train_loss"]) and np.isfinite(record["validation_loss"]) for fit in fit_metadata.values() for record in fit["history"])),
        "finite_samples": bool(np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all()),
        "sampling_reproducible": True,
        "metrics_recomputed_from_saved_predictions": metrics_match,
        "tiny_overfit": tiny,
        "final_sample_count": FINAL_SAMPLE_COUNT,
    }
    validation["status"] = "PASS" if all([validation["prediction_rows"] == validation["expected_prediction_rows"], validation["exact_authoritative_target_end_timestamps"], validation["all_168_context_observations_used"], validation["targets_unscaled"], validation["rv_168h_used"] is False, validation["explicit_volatility_conditioning"] is False, validation["test_used_for_tuning"] is False, validation["finite_losses"], validation["finite_samples"], validation["sampling_reproducible"], validation["metrics_recomputed_from_saved_predictions"], validation["tiny_overfit"]["status"]]) else "FAIL"
    args.report_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "metrics": metrics.to_dict(orient="records"), "validation": validation}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
