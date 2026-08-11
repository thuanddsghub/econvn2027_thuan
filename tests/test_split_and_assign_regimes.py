from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts.split_and_assign_regimes import split_and_assign


def test_boundaries_and_train_only_thresholds() -> None:
    timestamps = [
        datetime(2023, 12, 31, 22, tzinfo=UTC),
        datetime(2023, 12, 31, 23, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 31, 23, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 1, tzinfo=UTC),
    ]
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "rv_24h": [1.0, 2.0, 100.0, 100.0, 0.5, np.nan],
            "rv_168h": [999.0] * 6,
        }
    )

    result, thresholds = split_and_assign(frame)

    assert np.isclose(thresholds["q33"], 4 / 3)
    assert np.isclose(thresholds["q67"], 5 / 3)
    assert result["split"].tolist() == ["train", "train", "validation", "validation", "test", "test"]
    assert result["regime"].tolist() == ["Low", "High", "High", "High", "Low", pd.NA]


def test_nan_train_value_is_excluded_from_thresholds_and_rows_are_preserved() -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(4)],
            "rv_24h": [np.nan, 1.0, 2.0, 3.0],
        }
    )

    result, thresholds = split_and_assign(frame)

    assert len(result) == len(frame)
    assert np.isclose(thresholds["q33"], 5 / 3)
    assert result["regime"].isna().sum() == 1


def test_unsorted_input_fails_instead_of_reordering() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2023-12-31T23:00:00Z"],
            "rv_24h": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        split_and_assign(frame)
