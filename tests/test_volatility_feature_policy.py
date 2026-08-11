from pathlib import Path

import yaml


def test_main_config_selects_rv24_and_restricts_rv168() -> None:
    config = yaml.safe_load(Path("configs/base.yaml").read_text())

    assert config["data"]["primary_volatility_feature"] == "rv_24h"
    assert config["data"]["volatility_window"] == 24
    assert config["data"]["regime_threshold_feature"] == "rv_24h"
    assert config["data"]["regime_threshold_fit"] == "train_only"
    assert config["model"]["volatility_conditioning_feature"] == "rv_24h"
    assert config["experiment"]["primary_volatility_feature"] == "rv_24h"
    assert config["data"]["diagnostic_volatility_features"] == ["rv_168h"]
    assert config["experiment"]["diagnostic_volatility_features"] == ["rv_168h"]


def test_feature_metadata_marks_rv168_diagnostic_only() -> None:
    metadata = yaml.safe_load(
        Path("data/metadata/eth_usd_hourly_features_metadata.json").read_text()
    )

    assert metadata["primary_volatility_feature"] == "rv_24h"
    assert metadata["diagnostic_volatility_features"] == ["rv_168h"]
    policy = metadata["diagnostic_feature_policy"]
    assert "primary conditioning" in policy
    assert "regime thresholds" in policy
