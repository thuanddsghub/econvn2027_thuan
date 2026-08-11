import json
from datetime import datetime, timezone

from scripts.acquire_eth_usd_hourly import validate


def test_validation_reports_all_task02_conditions() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.11 compatibility
    end = datetime(2024, 1, 1, 4, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.11 compatibility
    rows = [
        [int(start.timestamp()), 100.0, 110.0, 105.0, 108.0, 2.0],
        [int((start.replace(hour=1)).timestamp()), 100.0, 110.0, 105.0, 108.0, 2.0],
        [int((start.replace(hour=1)).timestamp()), 100.0, 110.0, 105.0, 108.0, 2.0],
        [int((start.replace(hour=3)).timestamp()), 100.0, 110.0, 115.0, 108.0, -1.0],
    ]
    result = validate(rows, start, end)
    assert result["status"] == "FAIL"
    assert result["duplicate_timestamps"] == 2
    assert result["invalid_ohlc_rows"] == 1
    assert result["negative_volume_rows"] == 1
    assert result["nan_inf_rows"] == 0
    assert result["missing_hourly_candles"] == 1
    json.dumps(result)


def test_validation_passes_for_complete_valid_hourly_data() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.11 compatibility
    rows = [[int((start.replace(hour=i)).timestamp()), 100.0, 110.0, 105.0, 108.0, 2.0] for i in range(3)]
    result = validate(rows, start, datetime(2024, 1, 1, 3, tzinfo=timezone.utc))  # noqa: UP017 - Python 3.11 compatibility
    assert result["status"] == "PASS"
