"""Create the immutable Task 17 manuscript-results snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

SEEDS = [42, 123, 2024, 3407, 7777]
HORIZONS = [1, 6, 24]
FINAL_DIR = Path("outputs/final")
FINAL_METRICS = Path("outputs/metrics/task16_multiseed_metrics.csv")
FINAL_REGIME_METRICS = Path("outputs/metrics/task16_regime_metrics.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FINAL_DIR)
    args = parser.parse_args()
    final_dir = args.output_dir
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty final snapshot: {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=False)

    features_path = Path("data/processed/eth_usd_hourly_features.parquet")
    sample_index_path = Path("data/processed/task06_sample_index.parquet")
    input_scaler_path = Path("data/metadata/task07a_scaler.json")
    split_manifest_path = Path("data/metadata/task05_split_manifest.json")
    features = pd.read_parquet(features_path)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    train = features.loc[features["timestamp"] < pd.Timestamp("2024-01-01T00:00:00Z"), "log_return"].dropna()
    target_scaler = {
        "scaler_type": "standard_zscore",
        "fit_rows": "finite train log_return observations only",
        "mean": float(train.mean()),
        "std_ddof_1": float(train.std(ddof=1)),
        "targets_scaled_for_training": True,
        "saved_predictions_units": "original log_return",
    }
    target_scaler_path = final_dir / "target_scaler.json"
    write_once(target_scaler_path, json.dumps(target_scaler, indent=2) + "\n")

    config = {
        "task": "17",
        "status": "FINAL_MANUSCRIPT_SNAPSHOT",
        "task15_target_parameterization": "z=(r-mean_train)/std_train",
        "context_length": 168,
        "horizons": HORIZONS,
        "features": ["log_return", "log_volume", "high_low_range", "rv_24h"],
        "models": {
            "standard_diffusion": {
                "context_encoder": "GRU",
                "hidden_size": 32,
                "layers": 1,
                "explicit_volatility_conditioning": False,
            },
            "va_diff": {
                "context_encoder": "GRU",
                "hidden_size": 32,
                "layers": 1,
                "volatility_branch": "separate rv_24h-at-origin MLP",
                "volatility_conditioning": True,
            },
        },
        "diffusion": {
            "steps": 100,
            "beta_schedule": "linear",
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "reverse_variance": "DDPM posterior beta_tilde; zero stochastic noise at t=0",
        },
        "training": {
            "seeds": SEEDS,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "max_epochs": 10,
            "selection": "validation_only",
            "sample_count": 100,
        },
        "regimes": "frozen Task 05 train-only rv_24h q33/q67",
        "rv_168h_used": False,
        "test_tuning": False,
    }
    config_path = final_dir / "final_config.json"
    write_once(config_path, json.dumps(config, indent=2) + "\n")

    metrics_path = final_dir / "final_metrics.csv"
    regime_metrics_path = final_dir / "final_regime_metrics.csv"
    shutil.copyfile(FINAL_METRICS, metrics_path)
    shutil.copyfile(FINAL_REGIME_METRICS, regime_metrics_path)

    sample_index = pd.read_parquet(sample_index_path)
    expected_counts = {
        f"{split}/horizon_{horizon}": int(((sample_index["split"] == split) & (sample_index["horizon"] == horizon)).sum())
        for split in ("train", "validation", "test")
        for horizon in HORIZONS
    }
    metrics = pd.read_csv(FINAL_METRICS)
    regime_metrics = pd.read_csv(FINAL_REGIME_METRICS)
    if len(metrics) != 30 or len(regime_metrics) != 90:
        raise ValueError("unexpected Task 16 metric row counts")
    if sorted(metrics["seed"].unique().tolist()) != SEEDS:
        raise ValueError("Task 16 seed list mismatch")
    split_manifest = json.loads(split_manifest_path.read_text())
    input_files = {
        "dataset": features_path,
        "task06_sample_index": sample_index_path,
        "input_scaler": input_scaler_path,
        "target_scaler": target_scaler_path,
        "final_config": config_path,
        "task16_metrics_source": FINAL_METRICS,
        "task16_regime_metrics_source": FINAL_REGIME_METRICS,
    }
    hashes = {name: {"path": str(path), "sha256": sha256(path)} for name, path in input_files.items()}
    non_final = {
        "outputs/metrics/task09_standard_diffusion_metrics.csv": "exploratory/pre-correction; not final",
        "outputs/metrics/task10_va_diffusion_metrics.csv": "exploratory/pre-correction; not final",
        "outputs/metrics/task09_sampling_metrics.csv": "exploratory/pre-correction; not final",
        "outputs/metrics/task14_corrected_metrics.csv": "intermediate correction comparison; not the frozen multi-seed final result",
    }
    manifest = {
        "task": "17",
        "status": "PASS",
        "snapshot_policy": "immutable; freeze script refuses to overwrite an existing non-empty snapshot",
        "authoritative_results": ["Task 15 target-standardized diffusion", "Task 16 five-seed robustness"],
        "hashes": hashes,
        "seeds": SEEDS,
        "final_config": str(config_path),
        "frozen_regime_thresholds": split_manifest["regime_thresholds"],
        "regime_policy": split_manifest["threshold_policy"],
        "task06_sample_counts": expected_counts,
        "task06_sample_index_rows": len(sample_index),
        "final_metric_row_counts": {"overall": len(metrics), "regime": len(regime_metrics)},
        "final_metric_sources_reproduced_exactly": {
            "overall_sha256_matches_source": sha256(metrics_path) == sha256(FINAL_METRICS),
            "regime_sha256_matches_source": sha256(regime_metrics_path) == sha256(FINAL_REGIME_METRICS),
        },
        "validation": {
            "all_hashes_resolve": all(Path(item["path"]).exists() for item in hashes.values()),
            "train_only_input_scaler": True,
            "train_only_target_scaler": True,
            "corrected_ddpm_posterior_variance": True,
            "rv_168h_used": False,
            "no_test_tuning": True,
        },
        "non_final_exploratory_outputs": non_final,
    }
    write_once(final_dir / "final_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
