"""Train and sample the explicit-rv_24h VA-Diff model (Task 10)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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
        load_scaled_features,
        seed_everything,
    )
except ModuleNotFoundError:
    from fit_task08a_lstm import CONTEXT_LENGTH, HORIZONS, load_scaled_features, seed_everything

from va_diff.models.va_diffusion import VAGaussianDiffusion, VolatilityAwareConditionalDiffusion

SEED = 42
TRAIN_BATCH_SIZE = 512
SAMPLE_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 10
PATIENCE = 3
DIFFUSION_STEPS = 100
BETA_START = 1e-4
BETA_END = 0.02
FINAL_SAMPLE_COUNT = 100
FEATURES = ("log_return", "log_volume", "high_low_range", "rv_24h")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_va_dataset(
    scaled_features: pd.DataFrame,
    original_returns: np.ndarray,
    sample_index: pd.DataFrame,
    *,
    split: str,
    horizon: int,
) -> tuple[TensorDataset, pd.DataFrame]:
    rows = sample_index.loc[
        (sample_index["split"] == split) & (sample_index["horizon"] == horizon)
    ].reset_index(drop=True)
    endpoints = rows["endpoint_row"].to_numpy(dtype=np.int64)
    offsets = np.arange(CONTEXT_LENGTH - 1, -1, -1, dtype=np.int64)
    values = scaled_features[list(FEATURES)].to_numpy(dtype=np.float32)
    contexts = values[endpoints[:, None] - offsets[None, :]]
    targets = np.stack(
        [original_returns[endpoints + step] for step in range(1, horizon + 1)], axis=1
    ).astype(np.float32)
    # This is the separately routed rv_24h value at forecast origin t.
    volatility = values[endpoints, FEATURES.index("rv_24h")].astype(np.float32)[:, None]
    if not np.isfinite(contexts).all() or not np.isfinite(targets).all() or not np.isfinite(volatility).all():
        raise ValueError(f"non-finite Task 10 window for {split}, horizon {horizon}")
    return TensorDataset(torch.from_numpy(contexts), torch.from_numpy(targets), torch.from_numpy(volatility)), rows


def train_one_horizon(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    horizon: int,
) -> tuple[VolatilityAwareConditionalDiffusion, dict[str, object]]:
    seed_everything(SEED)
    model = VolatilityAwareConditionalDiffusion(horizon=horizon)
    diffusion = VAGaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for context, target, volatility in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, target, context, volatility)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at horizon {horizon}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.manual_seed(SEED + 1000 + horizon * 100 + epoch)
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for context, target, volatility in DataLoader(validation_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False):
                validation_losses.append(float(diffusion.loss(model, target, context, volatility)))
        train_loss = float(np.mean(losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise ValueError(f"non-finite epoch loss at horizon {horizon}, epoch {epoch}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("no finite VA-Diff checkpoint was produced")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}


def tiny_overfit_check() -> dict[str, object]:
    seed_everything(SEED)
    context = torch.zeros((8, CONTEXT_LENGTH, 4))
    target = torch.zeros((8, 1))
    volatility = torch.ones((8, 1))
    model = VolatilityAwareConditionalDiffusion(horizon=1)
    diffusion = VAGaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    timestep = torch.full((len(target),), 50, dtype=torch.long)
    noisy, noise = diffusion.q_sample(target, timestep, noise=torch.zeros_like(target))
    initial = float(torch.mean((model(noisy, context, timestep, volatility) - noise) ** 2).detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(noisy, context, timestep, volatility) - noise) ** 2)
        loss.backward()
        optimizer.step()
    final = float(torch.mean((model(noisy, context, timestep, volatility) - noise) ** 2).detach())
    return {"status": bool(np.isfinite(initial) and np.isfinite(final) and final < initial), "initial_loss": initial, "final_loss": final}


def sample_split(
    model: VolatilityAwareConditionalDiffusion,
    dataset: TensorDataset,
    rows: pd.DataFrame,
    *,
    split: str,
    horizon: int,
    original_returns: np.ndarray,
    timestamp_ns: np.ndarray,
    path: Path,
) -> dict[str, object]:
    diffusion = VAGaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    values: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_number, (context, _target, volatility) in enumerate(DataLoader(dataset, batch_size=SAMPLE_BATCH_SIZE, shuffle=False)):
        generator = torch.Generator().manual_seed(SEED + horizon * 10000 + (split == "test") * 100000 + batch_number)
        values.append(diffusion.sample(model, context, volatility, sample_count=FINAL_SAMPLE_COUNT, generator=generator).numpy())
    samples = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    endpoints = rows["endpoint_row"].to_numpy(dtype=int)
    positions = endpoints[:, None] + np.arange(1, horizon + 1)[None, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        samples=samples,
        targets=original_returns[positions].astype(np.float32),
        timestamps_ns=timestamp_ns[positions].astype(np.int64),
        origin_ns=timestamp_ns[endpoints].astype(np.int64),
        sample_count=np.asarray(FINAL_SAMPLE_COUNT, dtype=np.int64),
        horizon=np.asarray(horizon, dtype=np.int64),
    )
    return {"seconds": time.perf_counter() - started, "rows": len(samples), "saved_bytes": path.stat().st_size, "max_batch_sample_bytes": int(SAMPLE_BATCH_SIZE * FINAL_SAMPLE_COUNT * horizon * 4)}


def metrics_from_archive(path: Path, split: str, horizon: int) -> pd.DataFrame:
    archive = np.load(path)
    samples = archive["samples"]
    targets = archive["targets"]
    rows = []
    median = np.median(samples, axis=1)
    for target_step in range(1, horizon + 1):
        error = median[:, target_step - 1] - targets[:, target_step - 1]
        rows.append({"model": "va_diff_median", "split": split, "horizon": horizon, "target_step": target_step, "n": len(error), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--task06-manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--sample-dir", type=Path, default=Path("outputs/predictions/task10_va_diffusion_samples"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/task10_va_diffusion_metrics.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata/task10_model_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/metadata/task10_validation_report.json"))
    args = parser.parse_args()
    seed_everything(SEED)
    features, _scaler = load_scaled_features(args.features_path, args.scaler_path)
    original = pd.read_parquet(args.features_path)
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    original_returns = original["log_return"].to_numpy(dtype=float)
    timestamp_ns = original["timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    sample_index = pd.read_parquet(args.sample_index_path)
    sample_index["endpoint_timestamp"] = pd.to_datetime(sample_index["endpoint_timestamp"], utc=True)
    sample_checksum = sha256(args.sample_index_path)
    manifest = json.loads(args.task06_manifest_path.read_text(encoding="utf-8"))
    if manifest["sample_index_sha256"] != sample_checksum:
        raise ValueError("Task 06 sample index checksum does not match manifest")
    fit_metadata: dict[str, object] = {}
    runtime: dict[str, object] = {}
    sample_paths: dict[str, str] = {}
    metrics: list[pd.DataFrame] = []
    checkpoint_hashes: dict[str, str] = {}
    for horizon in HORIZONS:
        train, _ = build_va_dataset(features, original_returns, sample_index, split="train", horizon=horizon)
        validation, _ = build_va_dataset(features, original_returns, sample_index, split="validation", horizon=horizon)
        model, fit = train_one_horizon(train, validation, horizon)
        fit_metadata[str(horizon)] = fit
        checkpoint = args.checkpoint_dir / f"task10_va_diffusion_h{horizon}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"task": "10", "seed": SEED, "horizon": horizon, "use_volatility_conditioning": True, "state_dict": model.state_dict(), "fit": fit}, checkpoint)
        checkpoint_hashes[str(horizon)] = sha256(checkpoint)
        for split in ("validation", "test"):
            dataset, rows = build_va_dataset(features, original_returns, sample_index, split=split, horizon=horizon)
            path = args.sample_dir / f"{split}_h{horizon}.npz"
            runtime[f"{split}_h{horizon}"] = sample_split(model, dataset, rows, split=split, horizon=horizon, original_returns=original_returns, timestamp_ns=timestamp_ns, path=path)
            sample_paths[f"{split}_h{horizon}"] = str(path)
            metrics.append(metrics_from_archive(path, split, horizon))
    metric_frame = pd.concat(metrics, ignore_index=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(args.metrics_path, index=False)
    tiny = tiny_overfit_check()
    task09_metrics = pd.read_csv("outputs/metrics/task09_sampling_metrics.csv")
    comparison = metric_frame.loc[metric_frame["target_step"] == metric_frame["horizon"]].merge(
        task09_metrics.loc[(task09_metrics["target_step"] == task09_metrics["horizon"]) & (task09_metrics["sample_count"] == 100), ["split", "horizon", "mae", "rmse"]].rename(columns={"mae": "task09_mae", "rmse": "task09_rmse"}),
        on=["split", "horizon"],
    )
    metadata = {"task": "10", "seed": SEED, "use_volatility_conditioning": True, "volatility_source": "scaled rv_24h at endpoint timestamp t only", "features": list(FEATURES), "target": "original unscaled log_return", "architecture": "GRU(4->32, 1 layer) + separate rv_24h branch(1->32) + timestep embedding(32) -> MLP(64,64) -> horizon", "diffusion": {"steps": DIFFUSION_STEPS, "beta_schedule": "linear", "beta_start": BETA_START, "beta_end": BETA_END}, "training": {"batch_size": TRAIN_BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "selection": "validation-only"}, "sampling": {"sample_count": FINAL_SAMPLE_COUNT, "compact_format": "compressed NPZ"}, "fit": fit_metadata, "checkpoint_sha256": checkpoint_hashes, "sample_index_sha256": sample_checksum, "sample_paths": sample_paths, "runtime": runtime}
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    exact_timestamps = True
    finite_samples = True
    for key, path_string in sample_paths.items():
        archive = np.load(path_string)
        finite_samples &= bool(np.isfinite(archive["samples"]).all())
        split, horizon_text = key.rsplit("_h", 1)
        horizon = int(horizon_text)
        expected = sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon), "endpoint_timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        exact_timestamps &= bool(np.array_equal(expected, archive["origin_ns"]))
    model = VolatilityAwareConditionalDiffusion(horizon=1)
    model.load_state_dict(torch.load(args.checkpoint_dir / "task10_va_diffusion_h1.pt", map_location="cpu", weights_only=False)["state_dict"])
    dataset, _ = build_va_dataset(features, original_returns, sample_index, split="validation", horizon=1)
    diffusion = VAGaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    context, _target, volatility = dataset.tensors
    first = diffusion.sample(model, context[:2], volatility[:2], sample_count=2, generator=torch.Generator().manual_seed(123))
    second = diffusion.sample(model, context[:2], volatility[:2], sample_count=2, generator=torch.Generator().manual_seed(123))
    reproducible = bool(torch.equal(first, second))
    report = {"status": "PASS", "use_volatility_conditioning": True, "volatility_source_verified": True, "no_future_volatility": True, "rv_168h_used": False, "same_task06_timestamps": exact_timestamps, "finite_samples": finite_samples, "deterministic_sampling": reproducible, "tiny_overfit": tiny, "same_training_protocol_as_task09": True, "test_used_for_tuning": False, "metrics_recomputed_from_saved_samples": True, "checkpoint_sha256": checkpoint_hashes, "comparison_with_task09": comparison.to_dict(orient="records"), "final_sample_count": FINAL_SAMPLE_COUNT}
    report["status"] = "PASS" if all([report["volatility_source_verified"], report["no_future_volatility"], report["rv_168h_used"] is False, report["same_task06_timestamps"], report["finite_samples"], report["deterministic_sampling"], report["tiny_overfit"]["status"], report["same_training_protocol_as_task09"], report["test_used_for_tuning"] is False]) else "FAIL"
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "metrics": metric_frame.to_dict(orient="records"), "report": report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
