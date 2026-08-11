from datetime import UTC, datetime

import numpy as np
import pandas as pd

from scripts.preprocess_eth_usd_hourly import preprocess


def write_raw(path, timestamps, close=None, volume=None):
    close = close or tuple(100.0 + 10.0 * i for i in range(len(timestamps)))
    volume = volume or tuple(float(i) for i in range(len(timestamps)))
    pd.DataFrame(
        {
            "timestamp": [int(timestamp.timestamp()) for timestamp in timestamps],
            "low": [value - 1 for value in close],
            "high": [value + 1 for value in close],
            "open": close,
            "close": close,
            "volume": volume,
        }
    ).to_csv(path, index=False)


def test_returns_are_nan_after_gap_and_transforms_are_reference_values(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    raw_path = tmp_path / "raw.csv"
    write_raw(raw_path, [start, start.replace(hour=1), start.replace(hour=3)], volume=(0.0, 3.0, 8.0))

    processed, details = preprocess(raw_path)

    assert processed["timestamp"].dt.tz is not None
    assert np.isnan(processed.loc[0, "log_return"])
    assert np.isclose(processed.loc[1, "log_return"], np.log(110.0) - np.log(100.0))
    assert np.isnan(processed.loc[2, "log_return"])
    assert np.isclose(processed.loc[1, "log_volume"], np.log1p(3.0))
    assert np.isclose(processed.loc[1, "high_low_range"], 2.0 / 110.0)
    assert details["validation"]["returns_invalidated_because_of_gaps"] == 1
    assert details["validation"]["missing_hourly_candles"] == 1


def test_preprocessing_sorts_and_is_deterministic(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    raw_path = tmp_path / "raw.csv"
    write_raw(raw_path, [start.replace(hour=1), start])

    first, _ = preprocess(raw_path)
    second, _ = preprocess(raw_path)

    pd.testing.assert_frame_equal(first, second)
    assert first["timestamp"].is_monotonic_increasing
