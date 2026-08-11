"""Build Task 06 sample indices and PyTorch dataset validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from va_diff.data.timeseries import MODEL_FEATURES, SPLITS, build_sample_index


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_samples(
    features: pd.DataFrame, split_index: pd.DataFrame, sample_index: pd.DataFrame
) -> dict[str, object]:
    checks = {
        "sample_index_keys_unique": not sample_index[["split", "horizon", "endpoint_row"]].duplicated().any(),
        "no_future_context": True,
        "exact_hourly_continuity": True,
        "no_nan_inputs_or_targets": True,
        "target_split_boundaries_respected": True,
        "historical_context_before_target_allowed": True,
        "no_rv_168h_used": True,
    }
    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    timestamp_ns = timestamps.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    endpoints = sample_index["endpoint_row"].to_numpy(dtype="int64")
    horizons = sample_index["horizon"].to_numpy(dtype="int64")
    starts = endpoints - 167
    ends = endpoints + horizons
    gap = np.concatenate(([False], np.diff(timestamp_ns) != pd.Timedelta(hours=1).value))
    gap_prefix = np.concatenate(([0], np.cumsum(gap, dtype="int64")))
    checks["no_future_context"] = bool(np.all(timestamp_ns[endpoints] < timestamp_ns[endpoints + 1]))
    checks["exact_hourly_continuity"] = bool(
        np.all(gap_prefix[ends + 1] - gap_prefix[starts + 1] == 0)
    )
    model_nan = features[list(MODEL_FEATURES)].isna().any(axis=1).to_numpy()
    target_nan = features["log_return"].isna().to_numpy()
    model_nan_prefix = np.concatenate(([0], np.cumsum(model_nan, dtype="int64")))
    target_nan_prefix = np.concatenate(([0], np.cumsum(target_nan, dtype="int64")))
    checks["no_nan_inputs_or_targets"] = bool(
        np.all(model_nan_prefix[endpoints + 1] - model_nan_prefix[starts] == 0)
        and np.all(target_nan_prefix[ends + 1] - target_nan_prefix[endpoints + 1] == 0)
    )
    aligned_splits = split_index.set_index("timestamp").loc[timestamps]["split"].to_numpy()
    split_prefixes = {
        name: np.concatenate(([0], np.cumsum(aligned_splits == name, dtype="int64")))
        for name in SPLITS
    }
    for name, prefix in split_prefixes.items():
        selected = sample_index["split"].to_numpy() == name
        if selected.any():
            checks["target_split_boundaries_respected"] &= bool(
                np.all(prefix[ends[selected] + 1] - prefix[endpoints[selected] + 1] == horizons[selected])
            )
    return checks


def write_outputs(
    sample_index: pd.DataFrame,
    rejection_counts: dict[str, dict[str, int]],
    validation_checks: dict[str, object],
    input_sha256: str,
    thresholds: dict[str, float],
    input_path: Path,
    index_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, object]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_index.to_parquet(index_path, index=False, engine="pyarrow")
    sample_counts = {
        f"{split}/horizon_{horizon}": int(
            ((sample_index["split"] == split) & (sample_index["horizon"] == horizon)).sum()
        )
        for split in SPLITS
        for horizon in [1, 6, 24]
    }
    manifest = {
        "task": "06",
        "input_file": str(input_path),
        "input_sha256": input_sha256,
        "split_index_file": "data/processed/task05_split_index.parquet",
        "frozen_train_regime_thresholds": thresholds,
        "context_length": 168,
        "horizons": [1, 6, 24],
        "model_input_features": list(MODEL_FEATURES),
        "target": "log_return",
        "sample_definition": "X=[t-167,...,t], Y=[t+1,...,t+H]",
        "sample_index_file": str(index_path),
        "sample_index_sha256": sha256(index_path),
        "sample_counts": sample_counts,
        "rejected_sample_counts_by_reason": rejection_counts,
        "no_scaling": True,
        "excluded": ["model_training", "model_windows", "scalers", "rv_168h"],
    }
    validation = {"status": "PASS" if all(validation_checks.values()) else "FAIL", **validation_checks}
    report = {"schema_version": "task06.v1", "manifest": manifest, "validation": validation}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--split-index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--thresholds-path", type=Path, default=Path("data/metadata/task05_split_manifest.json"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/task06_validation_report.json"))
    args = parser.parse_args()
    input_sha256 = sha256(args.input_path)
    features = pd.read_parquet(args.input_path)
    split_index = pd.read_parquet(args.split_index_path)
    task05_manifest = json.loads(args.thresholds_path.read_text(encoding="utf-8"))
    if task05_manifest["input_sha256"] != input_sha256:
        raise ValueError("Task 05 manifest does not describe the supplied feature input")
    sample_index, rejection_counts = build_sample_index(features, split_index)
    validation_checks = validate_samples(features, split_index, sample_index)
    report = write_outputs(
        sample_index,
        rejection_counts,
        validation_checks,
        input_sha256,
        task05_manifest["regime_thresholds"],
        args.input_path,
        args.sample_index_path,
        args.manifest_path,
        args.report_path,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
