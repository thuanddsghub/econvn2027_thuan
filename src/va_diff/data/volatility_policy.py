"""Safety checks for selecting volatility features for experiments."""

from __future__ import annotations

import pandas as pd

PRIMARY_VOLATILITY_FEATURE = "rv_24h"
DIAGNOSTIC_VOLATILITY_FEATURES = frozenset({"rv_168h"})


class VolatilityFeatureSelectionError(ValueError):
    """Raised when a volatility feature violates the experiment policy."""


def validate_volatility_feature(
    feature_name: str = PRIMARY_VOLATILITY_FEATURE, *, sensitivity_mode: bool = False
) -> str:
    """Validate and return a volatility feature name for the requested mode.

    Main/default mode accepts only ``rv_24h``. ``rv_168h`` is available only when
    the caller explicitly enables ``sensitivity_mode`` for a diagnostic analysis.
    """
    if feature_name == PRIMARY_VOLATILITY_FEATURE:
        return feature_name
    if feature_name in DIAGNOSTIC_VOLATILITY_FEATURES and sensitivity_mode:
        return feature_name
    if feature_name in DIAGNOSTIC_VOLATILITY_FEATURES:
        raise VolatilityFeatureSelectionError(
            "rv_168h is diagnostic-only and cannot be selected in the main/default "
            "experiment; enable sensitivity_mode=True for an explicit diagnostic "
            "or sensitivity analysis."
        )
    raise VolatilityFeatureSelectionError(
        f"Unsupported volatility feature {feature_name!r}: main/default mode accepts "
        "only rv_24h."
    )


def select_volatility_feature(
    frame: pd.DataFrame,
    feature_name: str = PRIMARY_VOLATILITY_FEATURE,
    *,
    sensitivity_mode: bool = False,
) -> pd.Series:
    """Select a policy-approved volatility column from a feature frame."""
    validated_name = validate_volatility_feature(
        feature_name, sensitivity_mode=sensitivity_mode
    )
    if validated_name not in frame.columns:
        raise KeyError(f"volatility feature {validated_name!r} is absent from the feature frame")
    return frame[validated_name]
