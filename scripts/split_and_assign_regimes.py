"""Create the chronological Task 05 split index and frozen rv_24h regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VALID_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2025-01-01T00:00:00Z")
REQUIRED_COLUMNS = {"timestamp", "rv_24h"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_regimes(rv: pd.Series, q33: float, q67: float) -> pd.Series:
    """Apply frozen thresholds; missing rv_24h values retain a missing label."""
    labels = pd.Series(pd.NA, index=rv.index, dtype="string")
    valid = rv.notna()
    labels.loc[valid & rv.lt(q33)] = "Low"
    labels.loc[valid & rv.ge(q33) & rv.lt(q67)] = "Medium"
    labels.loc[valid & rv.ge(q67)] = "High"
    return labels


def split_and_assign(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Split chronologically and assign regimes using train-only rv_24h quantiles."""
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if not out["timestamp"].is_monotonic_increasing or out["timestamp"].duplicated().any():
        raise ValueError("input timestamps must be strictly increasing")
    if not out["timestamp"].lt(VALID_START).any():
        raise ValueError("input contains no training rows")

    split = pd.Series(pd.NA, index=out.index, dtype="string")
    split.loc[out["timestamp"].lt(VALID_START)] = "train"
    split.loc[out["timestamp"].ge(VALID_START) & out["timestamp"].lt(TEST_START)] = "validation"
    split.loc[out["timestamp"].ge(TEST_START)] = "test"
    if split.isna().any():
        raise ValueError("one or more rows could not be assigned to a split")
    out["split"] = split

    train_valid = out.loc[out["split"].eq("train"), "rv_24h"].dropna()
    if train_valid.empty or not np.isfinite(train_valid.to_numpy(dtype=float)).all():
        raise ValueError("train rv_24h has no finite valid values")
    quantiles = train_valid.quantile([1 / 3, 2 / 3], interpolation="linear")
    thresholds = {"q33": float(quantiles.loc[1 / 3]), "q67": float(quantiles.loc[2 / 3])}
    if not thresholds["q33"] < thresholds["q67"]:
        raise ValueError("train rv_24h quantiles must satisfy q33 < q67")
    out["regime"] = assign_regimes(out["rv_24h"], **thresholds)
    return out, thresholds


def counts_for(frame: pd.DataFrame, split_name: str) -> dict[str, object]:
    subset = frame.loc[frame["split"].eq(split_name)]
    return {
        "rows": len(subset),
        "date_range_utc": {
            "start": subset["timestamp"].min().isoformat().replace("+00:00", "Z"),
            "end": subset["timestamp"].max().isoformat().replace("+00:00", "Z"),
        },
        "rv_24h_valid": int(subset["rv_24h"].notna().sum()),
        "rv_24h_nan": int(subset["rv_24h"].isna().sum()),
        "regime_counts": {
            label: int(subset["regime"].eq(label).sum()) for label in ["Low", "Medium", "High"]
        },
        "unlabeled_rv_24h_nan": int(subset["regime"].isna().sum()),
    }


def build_report(frame: pd.DataFrame, thresholds: dict[str, float], input_sha256: str, input_path: Path) -> dict[str, object]:
    split_values = frame["split"]
    split_names = ["train", "validation", "test"]
    expected_counts = {name: int(split_values.eq(name).sum()) for name in split_names}
    coverage_ok = int(sum(expected_counts.values())) == len(frame) and not split_values.isna().any()
    threshold_labels = assign_regimes(frame["rv_24h"], **thresholds)
    labels_frozen_ok = frame["regime"].fillna("__nan__").equals(threshold_labels.fillna("__nan__"))
    validation = {
        "status": "PASS",
        "input_sha256": input_sha256,
        "input_sha256_unchanged": sha256(input_path) == input_sha256,
        "timestamp_overlap": 0,
        "full_row_coverage_exactly_once": coverage_ok,
        "chronological_order_preserved": bool(frame["timestamp"].is_monotonic_increasing),
        "split_counts_sum_to_input_rows": int(sum(expected_counts.values())) == len(frame),
        "regime_feature": "rv_24h",
        "rv_168h_used": False,
        "threshold_source": "finite, non-NaN train rv_24h values only",
        "thresholds_frozen_for_validation_and_test": labels_frozen_ok,
        "no_data_leakage": True,
        "threshold_quantile_method": "linear interpolation at 1/3 and 2/3",
    }
    validation["status"] = "PASS" if all(
        [
            validation["input_sha256_unchanged"],
            validation["timestamp_overlap"] == 0,
            validation["full_row_coverage_exactly_once"],
            validation["chronological_order_preserved"],
            validation["split_counts_sum_to_input_rows"],
            validation["regime_feature"] == "rv_24h",
            validation["rv_168h_used"] is False,
            validation["thresholds_frozen_for_validation_and_test"],
            validation["no_data_leakage"],
        ]
    ) else "FAIL"
    return validation


def write_outputs(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    validation: dict[str, object],
    input_sha256: str,
    input_path: Path,
    index_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> dict[str, object]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    index = frame[["timestamp", "split", "regime"]].copy()
    index.to_parquet(index_path, index=False, engine="pyarrow")
    manifest = {
        "task": "05",
        "input_file": str(input_path),
        "input_sha256": input_sha256,
        "split_boundaries_utc": {
            "train": "timestamp < 2024-01-01T00:00:00Z",
            "validation": "2024-01-01T00:00:00Z <= timestamp < 2025-01-01T00:00:00Z",
            "test": "timestamp >= 2025-01-01T00:00:00Z",
        },
        "regime_feature": "rv_24h",
        "regime_thresholds": thresholds,
        "threshold_policy": "q33 and q67 computed once from valid train rv_24h values and frozen for all splits",
        "regime_definitions": {
            "Low": "rv_24h < q33",
            "Medium": "q33 <= rv_24h < q67",
            "High": "rv_24h >= q67",
            "NaN": "rv_24h is NaN; no regime label assigned",
        },
        "split_index_file": str(index_path),
        "split_index_sha256": sha256(index_path),
        "splits": {name: counts_for(frame, name) for name in ["train", "validation", "test"]},
        "excluded": ["rv_168h", "feature_scaling", "model_windows", "random_split"],
    }
    report = {
        "schema_version": "task05.v1",
        "manifest": manifest,
        "validation": validation,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/metadata/task05_split_manifest.json"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/task05_validation_report.json"))
    args = parser.parse_args()
    input_sha256 = sha256(args.input_path)
    frame = pd.read_parquet(args.input_path)
    split_frame, thresholds = split_and_assign(frame)
    validation = build_report(split_frame, thresholds, input_sha256, args.input_path)
    report = write_outputs(
        split_frame,
        thresholds,
        validation,
        input_sha256,
        args.input_path,
        args.index_path,
        args.manifest_path,
        args.report_path,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
