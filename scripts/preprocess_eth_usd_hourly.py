"""Preprocess raw ETH/USD hourly OHLCV data for Task 03."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["timestamp", "low", "high", "open", "close", "volume"]
OUTPUT_COLUMNS = RAW_COLUMNS + ["log_return", "log_volume", "high_low_range"]
EXPECTED_INTERVAL = pd.Timedelta(hours=1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocess(raw_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    input_sha256 = sha256(raw_path)
    frame = pd.read_csv(raw_path)
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected raw columns: {list(frame.columns)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("raw data contains duplicate timestamps")

    time_delta = frame["timestamp"].diff()
    consecutive = time_delta.eq(EXPECTED_INTERVAL)
    frame["log_return"] = np.where(
        consecutive,
        np.log(frame["close"]) - np.log(frame["close"].shift(1)),
        np.nan,
    )
    frame["log_volume"] = np.log1p(frame["volume"])
    frame["high_low_range"] = (frame["high"] - frame["low"]) / frame["close"]

    gap_invalidated = int((~consecutive & frame.index.to_series().gt(0)).sum())
    missing_timestamps = []
    for previous, current in zip(frame["timestamp"].iloc[:-1], frame["timestamp"].iloc[1:]):
        if current - previous > EXPECTED_INTERVAL:
            missing_timestamps.extend(
                pd.date_range(previous + EXPECTED_INTERVAL, current - EXPECTED_INTERVAL, freq="h", tz="UTC")
            )
    validation = {
        "status": "PASS",
        "raw_sha256": input_sha256,
        "raw_sha256_unchanged": sha256(raw_path) == input_sha256,
        "processed_row_count": len(frame),
        "timestamps_utc": str(frame["timestamp"].dt.tz) == "UTC",
        "timestamps_strictly_increasing": bool(frame["timestamp"].is_monotonic_increasing and not frame["timestamp"].duplicated().any()),
        "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
        "missing_hourly_candles": len(missing_timestamps),
        "missing_candle_timestamps_utc": [stamp.isoformat().replace("+00:00", "Z") for stamp in missing_timestamps],
        "returns_invalidated_because_of_gaps": gap_invalidated,
        "returns_only_on_consecutive_hours": bool((frame.loc[~consecutive, "log_return"].isna()).all()),
        "unexpected_inf_counts": {
            column: int(np.isinf(frame[column].to_numpy(dtype=float)).sum())
            for column in ["log_return", "log_volume", "high_low_range"]
        },
        "nan_counts_by_column": {column: int(frame[column].isna().sum()) for column in OUTPUT_COLUMNS},
        "nan_causes": {
            "timestamp": "none",
            "source_ohlcv": "none expected after raw validation",
            "log_return": "first chronological row plus rows immediately after missing hourly intervals",
            "log_volume": "none expected after raw validation",
            "high_low_range": "none expected after raw validation",
        },
    }
    validation["status"] = "PASS" if all(
        [
            validation["raw_sha256_unchanged"],
            validation["timestamps_utc"],
            validation["timestamps_strictly_increasing"],
            validation["duplicate_timestamps"] == 0,
            validation["returns_only_on_consecutive_hours"],
            all(value == 0 for value in validation["unexpected_inf_counts"].values()),
        ]
    ) else "FAIL"
    return frame[OUTPUT_COLUMNS], {"validation": validation, "input_sha256": input_sha256}


def write_outputs(
    processed: pd.DataFrame,
    details: dict[str, object],
    output_path: Path,
    metadata_path: Path,
    report_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False, engine="pyarrow")
    output_sha256 = sha256(output_path)
    metadata = {
        "task": "03",
        "input_file": "data/raw/eth_usd_hourly_ohlcv.csv",
        "raw_sha256": details["input_sha256"],
        "output_file": str(output_path),
        "output_sha256": output_sha256,
        "row_count": len(processed),
        "columns": OUTPUT_COLUMNS,
        "transformations": {
            "timestamp": "parsed as UTC and sorted chronologically",
            "log_return": "log(close_t) - log(close_{t-1}) only when timestamps differ by exactly one hour",
            "log_volume": "numpy.log1p(volume)",
            "high_low_range": "(high - low) / close",
        },
        "excluded": ["realized_volatility", "volatility_regime", "data_split", "feature_scaling"],
    }
    report = {"schema_version": "task03.v1", "metadata": metadata, **details}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/eth_usd_hourly_ohlcv.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/eth_usd_hourly.parquet"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metadata/eth_usd_hourly_preprocessing_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/eth_usd_hourly_preprocessing_validation.json"))
    args = parser.parse_args()
    processed, details = preprocess(args.raw_path)
    report = write_outputs(processed, details, args.output_path, args.metadata_path, args.report_path)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
