import pandas as pd
import pytest

from va_diff.data.volatility_policy import (
    VolatilityFeatureSelectionError,
    select_volatility_feature,
    validate_volatility_feature,
)


def test_default_and_main_mode_allow_only_rv24h() -> None:
    assert validate_volatility_feature() == "rv_24h"
    assert validate_volatility_feature("rv_24h", sensitivity_mode=False) == "rv_24h"
    frame = pd.DataFrame({"rv_24h": [1.0], "rv_168h": [2.0]})
    pd.testing.assert_series_equal(
        select_volatility_feature(frame), frame["rv_24h"], check_names=True
    )


def test_rv168h_is_forbidden_without_explicit_sensitivity_mode() -> None:
    with pytest.raises(VolatilityFeatureSelectionError, match="diagnostic-only"):
        validate_volatility_feature("rv_168h")

    with pytest.raises(VolatilityFeatureSelectionError, match="sensitivity_mode=True"):
        select_volatility_feature(pd.DataFrame({"rv_168h": [2.0]}), "rv_168h")


def test_rv168h_is_allowed_only_with_explicit_sensitivity_mode() -> None:
    frame = pd.DataFrame({"rv_24h": [1.0], "rv_168h": [2.0]})
    assert validate_volatility_feature("rv_168h", sensitivity_mode=True) == "rv_168h"
    pd.testing.assert_series_equal(
        select_volatility_feature(frame, "rv_168h", sensitivity_mode=True),
        frame["rv_168h"],
        check_names=True,
    )


def test_unknown_feature_fails_fast_even_in_sensitivity_mode() -> None:
    with pytest.raises(VolatilityFeatureSelectionError, match="only rv_24h"):
        validate_volatility_feature("rv_1h", sensitivity_mode=True)
