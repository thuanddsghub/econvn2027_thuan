from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from va_diff.data.scaling import TargetScalingError, TrainOnlyStandardScaler


def test_train_only_zscore_and_inverse_transform() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "y": [5.0, 5.0, 5.0, 5.0]})
    scaler = TrainOnlyStandardScaler(("x", "y")).fit(train)

    transformed = scaler.transform_features(train)
    assert np.isclose(scaler.mean_["x"], 2.0)
    assert np.isclose(scaler.scale_["x"], np.sqrt(2 / 3))
    assert np.isnan(transformed.loc[3, "x"])
    assert np.allclose(scaler.inverse_transform_features(transformed).loc[:2, ["x", "y"]], train.loc[:2, ["x", "y"]])
    assert scaler.scale_["y"] == 1.0


def test_validation_values_do_not_change_fitted_parameters() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = TrainOnlyStandardScaler(("x",)).fit(train)
    before = scaler.to_dict()
    scaler.transform_features(pd.DataFrame({"x": [1000.0, 2000.0]}))
    assert scaler.to_dict() == before


def test_target_scaling_fails_fast_and_targets_remain_unscaled() -> None:
    scaler = TrainOnlyStandardScaler(("log_return",)).fit(pd.DataFrame({"log_return": [1.0, 2.0]}))

    with pytest.raises(TargetScalingError, match="Target scaling is disabled"):
        scaler.transform_target(np.array([1.0, 2.0]))
    with pytest.raises(TargetScalingError, match="original log-return scale"):
        scaler.inverse_transform_target(np.array([0.0, 1.0]))


def test_default_config_keeps_targets_unscaled() -> None:
    config = yaml.safe_load(Path("configs/base.yaml").read_text())
    assert config["training"]["targets_scaled"] is False
