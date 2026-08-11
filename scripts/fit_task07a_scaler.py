"""Fit and validate a train-only Task 07A feature scaler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from va_diff.data.scaling import TrainOnlyStandardScaler

FEATURES = ("log_return", "log_volume", "high_low_range", "rv_24h")
VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2025-01-01T00:00:00Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    return {
        "train": frame.loc[timestamp < VALIDATION_START].copy(),
        "validation": frame.loc[(timestamp >= VALIDATION_START) & (timestamp < TEST_START)].copy(),
        "test": frame.loc[timestamp >= TEST_START].copy(),
    }


def transformed_summary(frame: pd.DataFrame, scaler: TrainOnlyStandardScaler) -> dict[str, object]:
    transformed = scaler.transform_features(frame)
    summary: dict[str, object] = {}
    for feature in FEATURES:
        values = transformed[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary[feature] = {
            "finite_count": int(finite.size),
            "nan_count": int(np.isnan(values).sum()),
            "mean": float(np.mean(finite)) if finite.size else None,
            "std_population": float(np.std(finite, ddof=0)) if finite.size else None,
        }
    return summary


def inverse_error(frame: pd.DataFrame, scaler: TrainOnlyStandardScaler) -> float:
    finite_rows = frame[list(FEATURES)].notna().all(axis=1)
    reference = frame.loc[finite_rows].iloc[[0, len(frame.loc[finite_rows]) // 2, -1]]
    reconstructed = scaler.inverse_transform_features(scaler.transform_features(reference))
    return float(np.max(np.abs(reconstructed[list(FEATURES)].to_numpy() - reference[list(FEATURES)].to_numpy())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/task07a_validation_report.json"))
    args = parser.parse_args()

    input_sha256 = sha256(args.input_path)
    sample_index_sha256 = sha256(args.sample_index_path)
    features = pd.read_parquet(args.input_path)
    sample_index = pd.read_parquet(args.sample_index_path)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    sample_index["endpoint_timestamp"] = pd.to_datetime(sample_index["endpoint_timestamp"], utc=True)
    if "rv_168h" not in features.columns or not set(FEATURES).issubset(features.columns):
        raise ValueError("Task 07A input does not contain the required feature columns")
    if not sample_index["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Task 06 sample index contains unknown split labels")

    splits = split_rows(features)
    scaler = TrainOnlyStandardScaler(FEATURES).fit(splits["train"])
    summaries = {name: transformed_summary(frame, scaler) for name, frame in splits.items()}
    nan_preserved = all(
        int(np.isnan(scaler.transform_features(frame)[feature].to_numpy(dtype=float)).sum())
        == int(np.isnan(frame[feature].to_numpy(dtype=float)).sum())
        for frame in splits.values()
        for feature in FEATURES
    )
    no_inf = all(
        not np.isinf(scaler.transform_features(frame)[list(FEATURES)].to_numpy(dtype=float)).any()
        for frame in splits.values()
    )
    validation = {
        "status": "PASS",
        "input_sha256": input_sha256,
        "sample_index_sha256": sample_index_sha256,
        "input_sha256_unchanged": sha256(args.input_path) == input_sha256,
        "sample_index_unchanged": sha256(args.sample_index_path) == sample_index_sha256,
        "scaler_type": "StandardScaler-equivalent z-score",
        "fit_source": "finite observations from rows timestamp < 2024-01-01T00:00:00Z only",
        "validation_test_contributed_to_fit": False,
        "rv_168h_used": False,
        "targets_scaled": False,
        "nan_counts_preserved": nan_preserved,
        "no_inf_introduced": no_inf,
        "inverse_transform_max_abs_error": {
            name: inverse_error(frame, scaler) for name, frame in splits.items()
        },
        "constant_variance_policy": "variance <= 1e-12 uses scale=1.0",
    }
    validation["status"] = "PASS" if all(
        [
            validation["input_sha256_unchanged"],
            validation["sample_index_unchanged"],
            not validation["validation_test_contributed_to_fit"],
            not validation["rv_168h_used"],
            not validation["targets_scaled"],
            validation["nan_counts_preserved"],
            validation["no_inf_introduced"],
            all(error < 1e-12 for error in validation["inverse_transform_max_abs_error"].values()),
        ]
    ) else "FAIL"
    scaler_payload = {
        "task": "07A",
        "input_file": str(args.input_path),
        "input_sha256": input_sha256,
        "sample_index_file": str(args.sample_index_path),
        "sample_index_sha256": sample_index_sha256,
        "split_boundaries_utc": {
            "train": "timestamp < 2024-01-01T00:00:00Z",
            "validation": "2024-01-01T00:00:00Z <= timestamp < 2025-01-01T00:00:00Z",
            "test": "timestamp >= 2025-01-01T00:00:00Z",
        },
        "features": list(FEATURES),
        "fit_policy": "fit finite values from train rows only; preserve NaNs; targets are not scaled",
        "scaler": scaler.to_dict(),
        "transformed_summary": summaries,
    }
    args.scaler_path.parent.mkdir(parents=True, exist_ok=True)
    args.scaler_path.write_text(json.dumps(scaler_payload, indent=2) + "\n", encoding="utf-8")
    args.report_path.write_text(json.dumps({"schema_version": "task07a.v1", "validation": validation}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scaler": scaler_payload, "validation": validation}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
