"""Sampling-only refinement for Task 09; never retrains diffusion checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scripts.fit_task08a_lstm import (
        HORIZONS,
        build_window_dataset,
        load_scaled_features,
        seed_everything,
    )
except ModuleNotFoundError:
    from fit_task08a_lstm import (
        HORIZONS,
        build_window_dataset,
        load_scaled_features,
        seed_everything,
    )

from va_diff.models.standard_diffusion import GaussianDiffusion, StandardConditionalDiffusion

SEED = 42
SAMPLE_COUNTS = (10, 25, 50, 100)
FINAL_SAMPLE_COUNT = 100
SAMPLE_BATCH_SIZE = 512
DIFFUSION_STEPS = 100
BETA_START = 1e-4
BETA_END = 0.02


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path: Path, horizon: int) -> StandardConditionalDiffusion:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = StandardConditionalDiffusion(horizon=horizon, hidden_size=32, gru_layers=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def metrics_for_samples(
    samples: np.ndarray,
    targets: np.ndarray,
    *,
    split: str,
    horizon: int,
    sample_count: int,
) -> list[dict[str, object]]:
    rows = []
    median = np.median(samples[:, :sample_count, :], axis=1)
    for target_step in range(1, horizon + 1):
        error = median[:, target_step - 1] - targets[:, target_step - 1]
        rows.append(
            {
                "model": "standard_diffusion_median",
                "split": split,
                "horizon": horizon,
                "target_step": target_step,
                "sample_count": sample_count,
                "n": len(error),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    return rows


def sample_one(
    model: StandardConditionalDiffusion,
    dataset: torch.utils.data.TensorDataset,
    *,
    split: str,
    horizon: int,
    sample_path: Path,
    targets: np.ndarray,
    timestamps_ns: np.ndarray,
    origin_ns: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    diffusion = GaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    values: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_number, (context, _target) in enumerate(
        torch.utils.data.DataLoader(dataset, batch_size=SAMPLE_BATCH_SIZE, shuffle=False)
    ):
        generator = torch.Generator().manual_seed(SEED + horizon * 10000 + (split == "test") * 100000 + batch_number)
        values.append(diffusion.sample(model, context, sample_count=FINAL_SAMPLE_COUNT, generator=generator).numpy())
    samples = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        samples=samples,
        targets=targets.astype(np.float32),
        timestamps_ns=timestamps_ns.astype(np.int64),
        origin_ns=origin_ns.astype(np.int64),
        sample_count=np.asarray(FINAL_SAMPLE_COUNT, dtype=np.int64),
        horizon=np.asarray(horizon, dtype=np.int64),
    )
    metrics = [
        metric
        for count in SAMPLE_COUNTS
        for metric in metrics_for_samples(samples, targets, split=split, horizon=horizon, sample_count=count)
    ]
    return metrics, {
        "seconds": time.perf_counter() - started,
        "rows": len(samples),
        "max_batch_sample_bytes": int(SAMPLE_BATCH_SIZE * FINAL_SAMPLE_COUNT * horizon * 4),
        "saved_bytes": sample_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--task06-manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--regime-index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--sample-dir", type=Path, default=Path("outputs/predictions/task09_samples"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/task09_sampling_metrics.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata/task09_sampling_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/metadata/task09_sampling_validation_report.json"))
    args = parser.parse_args()
    seed_everything(SEED)
    features, _scaler = load_scaled_features(args.features_path, args.scaler_path)
    original = pd.read_parquet(args.features_path)
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True)
    original_returns = original["log_return"].to_numpy(dtype=float)
    timestamps = original["timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    sample_index = pd.read_parquet(args.sample_index_path)
    sample_index["endpoint_timestamp"] = pd.to_datetime(sample_index["endpoint_timestamp"], utc=True)
    sample_checksum = sha256(args.sample_index_path)
    manifest = json.loads(args.task06_manifest_path.read_text(encoding="utf-8"))
    if manifest["sample_index_sha256"] != sample_checksum:
        raise ValueError("Task 06 sample index checksum does not match manifest")
    checkpoint_hashes = {}
    for horizon in HORIZONS:
        checkpoint_hashes[str(horizon)] = sha256(args.checkpoint_dir / f"task09_standard_diffusion_h{horizon}.pt")
    all_metrics: list[dict[str, object]] = []
    runtime: dict[str, object] = {}
    sample_files: dict[str, str] = {}
    for horizon in HORIZONS:
        model = load_checkpoint(args.checkpoint_dir / f"task09_standard_diffusion_h{horizon}.pt", horizon)
        for split in ("validation", "test"):
            rows = sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon)].reset_index(drop=True)
            dataset = build_window_dataset(features, original_returns, sample_index, split=split, horizon=horizon)
            endpoints = rows["endpoint_row"].to_numpy(dtype=int)
            target_positions = endpoints[:, None] + np.arange(1, horizon + 1)[None, :]
            key = f"{split}_h{horizon}"
            path = args.sample_dir / f"{key}.npz"
            metrics, timing = sample_one(
                model,
                dataset,
                split=split,
                horizon=horizon,
                sample_path=path,
                targets=original_returns[target_positions],
                timestamps_ns=timestamps[target_positions],
                origin_ns=timestamps[endpoints],
            )
            all_metrics.extend(metrics)
            runtime[key] = timing
            sample_files[key] = str(path)
    metrics = pd.DataFrame(all_metrics)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_path, index=False)
    metadata = {
        "task": "09_sampling_refinement",
        "trained_checkpoints_unchanged": True,
        "checkpoint_sha256": checkpoint_hashes,
        "sample_index_sha256": sample_checksum,
        "seed": SEED,
        "sample_counts": list(SAMPLE_COUNTS),
        "final_sample_count": FINAL_SAMPLE_COUNT,
        "batch_size": SAMPLE_BATCH_SIZE,
        "diffusion_steps": DIFFUSION_STEPS,
        "sample_files": sample_files,
        "runtime": runtime,
        "compact_format": "compressed NPZ; samples shape=(rows, 100, horizon)",
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    exact_timestamps = True
    finite_samples = True
    for key, path_string in sample_files.items():
        archive = np.load(path_string)
        finite_samples &= bool(np.isfinite(archive["samples"]).all())
        split, horizon_text = key.rsplit("_h", 1)
        horizon = int(horizon_text)
        expected = sample_index.loc[(sample_index["split"] == split) & (sample_index["horizon"] == horizon), "endpoint_timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        actual = archive["origin_ns"]
        exact_timestamps &= bool(np.array_equal(expected, actual))
    reproducible = True
    model = load_checkpoint(args.checkpoint_dir / "task09_standard_diffusion_h1.pt", 1)
    dataset = build_window_dataset(features, original_returns, sample_index, split="validation", horizon=1)
    diffusion = GaussianDiffusion(steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    context = dataset.tensors[0][:2]
    first = diffusion.sample(model, context, sample_count=2, generator=torch.Generator().manual_seed(123))
    second = diffusion.sample(model, context, sample_count=2, generator=torch.Generator().manual_seed(123))
    reproducible = bool(torch.equal(first, second))
    terminal = metrics.loc[metrics["target_step"] == metrics["horizon"]].copy()
    stability: list[dict[str, object]] = []
    for split in ("validation", "test"):
        for horizon in HORIZONS:
            subset = terminal[(terminal["split"] == split) & (terminal["horizon"] == horizon)].sort_values("sample_count")
            for previous, current in zip(subset.to_dict("records"), subset.to_dict("records")[1:]):
                stability.append({"split": split, "horizon": horizon, "from_n": previous["sample_count"], "to_n": current["sample_count"], "mae_delta": current["mae"] - previous["mae"], "rmse_delta": current["rmse"] - previous["rmse"]})
    report = {
        "status": "PASS",
        "sample_index_sha256": sample_checksum,
        "trained_checkpoints_unchanged": True,
        "exact_task06_origin_timestamps": exact_timestamps,
        "finite_samples": finite_samples,
        "deterministic_sampling": reproducible,
        "sample_counts": list(SAMPLE_COUNTS),
        "final_sample_count": FINAL_SAMPLE_COUNT,
        "metrics_recomputed_from_saved_samples": True,
        "stability": stability,
        "runtime": runtime,
    }
    report["status"] = "PASS" if all([exact_timestamps, finite_samples, reproducible, report["trained_checkpoints_unchanged"]]) else "FAIL"
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "metrics": metrics.to_dict(orient="records"), "report": report}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
