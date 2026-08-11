from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from scripts.compute_realized_volatility import add_features, compute_realized_volatility


def toy_frame(log_returns: list[float]) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(len(log_returns))],
            "log_return": log_returns,
        }
    )


def test_strict_trailing_window_matches_hand_calculation() -> None:
    frame = toy_frame([np.nan, 1.0, 2.0, 3.0, 4.0])
    result = compute_realized_volatility(frame, window=3)

    assert result.iloc[:3].isna().all()
    assert np.isclose(result.iloc[3], np.sqrt(1.0**2 + 2.0**2 + 3.0**2))
    assert np.isclose(result.iloc[4], np.sqrt(2.0**2 + 3.0**2 + 4.0**2))


def test_nan_return_propagates_through_strict_window_and_no_future_leakage() -> None:
    frame = toy_frame([np.nan, 1.0, np.nan, 3.0, 4.0, 5.0])
    result = add_features(frame)

    assert result["rv_24h"].isna().all()
    changed = frame.copy()
    changed.loc[5, "log_return"] = 500.0
    changed_result = add_features(changed)
    pd.testing.assert_series_equal(result.loc[:4, "rv_24h"], changed_result.loc[:4, "rv_24h"])
