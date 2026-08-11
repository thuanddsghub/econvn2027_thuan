"""Transparent train-only standard scaling for Task 07A."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TargetScalingError(ValueError):
    """Raised when the feature scaler is used on forecast targets."""


class TrainOnlyStandardScaler:
    """Feature-wise z-score scaler fitted on finite train observations only."""

    def __init__(self, features: tuple[str, ...], *, variance_epsilon: float = 1e-12) -> None:
        self.features = features
        self.variance_epsilon = variance_epsilon
        self.mean_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        self.variance_: dict[str, float] = {}
        self.n_train_observations_: dict[str, int] = {}

    def fit(self, train_frame: pd.DataFrame) -> TrainOnlyStandardScaler:
        for feature in self.features:
            values = train_frame[feature].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"infinite train value found in {feature}")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"no finite train observations found in {feature}")
            mean = float(np.mean(finite))
            variance = float(np.var(finite, ddof=0))
            self.mean_[feature] = mean
            self.variance_[feature] = variance
            self.scale_[feature] = 1.0 if variance <= self.variance_epsilon else float(np.sqrt(variance))
            self.n_train_observations_[feature] = int(finite.size)
        return self

    def _check_fitted(self) -> None:
        if set(self.mean_) != set(self.features):
            raise RuntimeError("scaler must be fitted before transform")

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = frame.copy()
        for feature in self.features:
            values = out[feature].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"infinite value found in {feature}")
            out[feature] = (values - self.mean_[feature]) / self.scale_[feature]
        return out

    def inverse_transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = frame.copy()
        for feature in self.features:
            values = out[feature].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"infinite value found in {feature}")
            out[feature] = values * self.scale_[feature] + self.mean_[feature]
        return out

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible alias for explicitly feature-only transformation."""
        return self.transform_features(frame)

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible alias for explicitly feature-only inversion."""
        return self.inverse_transform_features(frame)

    def transform_target(self, target: object) -> object:
        """Reject target scaling; forecasting targets remain original log returns."""
        raise TargetScalingError(
            "Target scaling is disabled in the default experiment: pass targets through "
            "unchanged and use transform_features() only for model input features."
        )

    def inverse_transform_target(self, target: object) -> object:
        """Reject target inverse-scaling for the same target-scale policy."""
        raise TargetScalingError(
            "Target inverse-scaling is disabled: forecasting targets remain in original "
            "log-return scale."
        )

    def to_dict(self) -> dict[str, object]:
        self._check_fitted()
        return {
            "scaler_type": "standard_zscore",
            "features": list(self.features),
            "variance_epsilon": self.variance_epsilon,
            "mean": self.mean_,
            "variance": self.variance_,
            "scale": self.scale_,
            "n_train_observations": self.n_train_observations_,
            "targets_scaled": False,
        }
