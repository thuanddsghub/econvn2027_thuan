"""Acquire and validate raw ETH/USD hourly candles from Coinbase Exchange."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

API_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
INTERVAL_SECONDS = 3600
MAX_CANDLES_PER_REQUEST = 300
RAW_COLUMNS = ["timestamp", "low", "high", "open", "close", "volume"]


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.11 compatibility
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return parsed.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)  # noqa: UP017 - Python 3.11 compatibility


def fetch_page(start: datetime, end: datetime) -> list[list[float]]:
    query = urlencode({"start": start.isoformat(), "end": end.isoformat(), "granularity": INTERVAL_SECONDS})
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": "va-diff-eth-task02/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Coinbase candles request failed for {start} to {end}: {exc}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, list) or len(row) != 6 for row in payload):
        raise RuntimeError(f"Unexpected Coinbase candles response for {start} to {end}")
    return payload


def acquire(start: datetime, end: datetime, pause_seconds: float = 0.15) -> list[list[float]]:
    if end <= start:
        raise ValueError("end must be after start")
    rows: list[list[float]] = []
    cursor = start
    page_span = timedelta(seconds=INTERVAL_SECONDS * MAX_CANDLES_PER_REQUEST)
    while cursor < end:
        page_end = min(cursor + page_span, end)
        rows.extend(fetch_page(cursor, page_end))
        cursor = page_end
        if cursor < end:
            time.sleep(pause_seconds)
    # Coinbase includes boundary candles in adjacent pages. Keep the first source row
    # for each timestamp so the saved raw dataset has one row per candle.
    return list({int(row[0]): row for row in reversed(rows)}.values())


def validate(rows: list[list[float]], start: datetime, end: datetime) -> dict[str, object]:
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    numeric = ["low", "high", "open", "close", "volume"]
    non_finite_mask = ~np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1) if not frame.empty else np.array([], dtype=bool)
    duplicate_count = int(frame["timestamp"].duplicated(keep=False).sum()) if not frame.empty else 0
    invalid_ohlc_mask = (
        ~(
            (frame["low"] <= frame["open"])
            & (frame["low"] <= frame["close"])
            & (frame["open"] <= frame["high"])
            & (frame["close"] <= frame["high"])
            & (frame["low"] <= frame["high"])
        )
        if not frame.empty
        else pd.Series(dtype=bool)
    )
    negative_volume_mask = frame["volume"] < 0 if not frame.empty else pd.Series(dtype=bool)
    timestamps = pd.to_datetime(frame["timestamp"], unit="s", utc=True) if not frame.empty else pd.DatetimeIndex([])
    expected = pd.date_range(start=start, end=end - timedelta(seconds=INTERVAL_SECONDS), freq="h", tz="UTC")
    observed = pd.DatetimeIndex(timestamps).drop_duplicates().sort_values()
    missing = expected.difference(observed)
    return {
        "status": "PASS" if not (duplicate_count or invalid_ohlc_mask.sum() or negative_volume_mask.sum() or non_finite_mask.sum()) else "FAIL",
        "row_count": len(frame),
        "duplicate_timestamps": duplicate_count,
        "invalid_ohlc_rows": int(invalid_ohlc_mask.sum()),
        "negative_volume_rows": int(negative_volume_mask.sum()),
        "nan_inf_rows": int(non_finite_mask.sum()),
        "missing_hourly_candles": len(missing),
        "missing_candle_timestamps_utc": [stamp.isoformat().replace("+00:00", "Z") for stamp in missing],
    }


def write_outputs(rows: list[list[float]], start: datetime, end: datetime, retrieval_time: str, raw_path: Path, metadata_path: Path, report_path: Path) -> dict[str, object]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    unique_rows = {int(row[0]): row for row in reversed(rows)}
    frame_rows = sorted(unique_rows.values(), key=lambda row: row[0])
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(RAW_COLUMNS)
        writer.writerows(frame_rows)
    checksum = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    validation = validate(frame_rows, start, end)
    metadata = {
        "source": "Coinbase Exchange public REST API",
        "source_url": API_URL,
        "retrieval_time_utc": retrieval_time,
        "date_range": {"start_utc": start.isoformat().replace("+00:00", "Z"), "end_utc_exclusive": end.isoformat().replace("+00:00", "Z")},
        "interval": "1h",
        "symbol": "ETH-USD",
        "row_count": len(frame_rows),
        "sha256": checksum,
        "raw_file": str(raw_path),
        "validation_report": str(report_path),
    }
    report = {"schema_version": "task02.v1", "dataset": metadata, "validation": validation}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-11T00:00:00Z", help="UTC end (exclusive); use a completed hourly boundary")
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/eth_usd_hourly_ohlcv.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metadata/eth_usd_hourly_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("data/metadata/eth_usd_hourly_validation.json"))
    args = parser.parse_args()
    start, end = parse_utc(args.start), parse_utc(args.end)
    retrieval_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017 - Python 3.11 compatibility
    rows = acquire(start, end, args.pause_seconds)
    report = write_outputs(rows, start, end, retrieval_time, args.raw_path, args.metadata_path, args.report_path)
    print(json.dumps(report, indent=2))
    return 0 if report["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
