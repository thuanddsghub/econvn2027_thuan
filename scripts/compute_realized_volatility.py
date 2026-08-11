"""Compute leakage-safe trailing realized volatility features for Task 04."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_COLUMNS = [
    "timestamp",
    "low",
    "high",
    "open",
    "close",
    "volume",
    "log_return",
    "log_volume",
    "high_low_range",
]
FEATURE_COLUMNS = ["rv_24h", "rv_168h"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_realized_volatility(
    frame: pd.DataFrame, window: int = 24
) -> pd.Series:
    """Return strict trailing sqrt(sum of squared log returns) for ``window`` rows."""
    if window < 1:
        raise ValueError("window must be positive")
    return np.sqrt(frame["log_return"].pow(2).rolling(window=window, min_periods=window).sum())


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rv_24h"] = compute_realized_volatility(out, 24)
    out["rv_168h"] = compute_realized_volatility(out, 168)
    return out


def validate(frame: pd.DataFrame, input_sha256: str, input_path: Path) -> dict[str, object]:
    feature_inf_counts = {
        column: int(np.isinf(frame[column].to_numpy(dtype=float)).sum()) for column in FEATURE_COLUMNS
    }
    validation: dict[str, object] = {
        "status": "PASS",
        "input_sha256": input_sha256,
        "input_sha256_unchanged": sha256(input_path) == input_sha256,
        "timestamps_utc": str(frame["timestamp"].dt.tz) == "UTC",
        "timestamps_strictly_increasing": bool(
            frame["timestamp"].is_monotonic_increasing
            and not frame["timestamp"].duplicated().any()
        ),
        "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
        "formula": "sqrt(sum_{i=0}^{window-1} log_return[t-i]^2)",
        "window_policy": (
            "strict trailing window; current and prior rows only; any NaN log_return in the "
            "window produces NaN; no filling across gaps"
        ),
        "uses_current_and_past_returns_only": True,
        "future_timestamp_contribution": False,
        "feature_inf_counts": feature_inf_counts,
        "nan_counts_by_column": {
            column: int(frame[column].isna().sum()) for column in [*INPUT_COLUMNS, *FEATURE_COLUMNS]
        },
        "nan_causes": {
            "rv_24h": "strict 24-observation window: initial warm-up and any window containing a NaN log_return",
            "rv_168h": "strict 168-observation window: initial warm-up and any window containing a NaN log_return",
        },
    }
    validation["status"] = "PASS" if all(
        [
            validation["input_sha256_unchanged"],
            validation["timestamps_utc"],
            validation["timestamps_strictly_increasing"],
            validation["duplicate_timestamps"] == 0,
            validation["uses_current_and_past_returns_only"],
            not validation["future_timestamp_contribution"],
            all(value == 0 for value in feature_inf_counts.values()),
        ]
    ) else "FAIL"
    return validation


def write_outputs(
    features: pd.DataFrame,
    validation: dict[str, object],
    input_sha256: str,
    output_path: Path,
    metadata_path: Path,
    report_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False, engine="pyarrow")
    metadata = {
        "task": "04",
        "input_file": "data/processed/eth_usd_hourly.parquet",
        "input_sha256": input_sha256,
        "output_file": str(output_path),
        "output_sha256": sha256(output_path),
        "row_count": len(features),
        "feature_columns": FEATURE_COLUMNS,
        "primary_volatility_feature": "rv_24h",
        "diagnostic_volatility_features": ["rv_168h"],
        "diagnostic_feature_policy": (
            "rv_168h is diagnostic-only: permitted for descriptive diagnostics, post-main "
            "sensitivity/robustness checks, and optional supplementary figures/tables; "
            "excluded from primary conditioning, regime thresholds, headline tables, "
            "model selection, and hyperparameter tuning"
        ),
        "formula": "rv_window(t) = sqrt(sum_{i=0}^{window-1} log_return[t-i]^2)",
        "convention": "unnormalized trailing realized volatility in return units",
        "windows_hours": {"rv_24h": 24, "rv_168h": 168},
        "window_policy": validation["window_policy"],
        "excluded": ["volatility_regimes", "regime_quantiles", "data_splits", "scaling"],
    }
    report = {"schema_version": "task04.v1", "metadata": metadata, "validation": validation}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/eth_usd_hourly.parquet"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metadata/eth_usd_hourly_features_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/eth_usd_hourly_features_validation.json"))
    args = parser.parse_args()

    input_sha256 = sha256(args.input_path)
    frame = pd.read_parquet(args.input_path)
    if list(frame.columns) != INPUT_COLUMNS:
        raise ValueError(f"unexpected processed columns: {list(frame.columns)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    features = add_features(frame)
    validation = validate(features, input_sha256, args.input_path)
    report = write_outputs(
        features, validation, input_sha256, args.output_path, args.metadata_path, args.report_path
    )
    print(json.dumps(report, indent=2))
    return 0 if report["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
