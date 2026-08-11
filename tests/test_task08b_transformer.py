import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fit_task08a_lstm import seed_everything
from scripts.fit_task08b_transformer import (
    TransformerForecaster,
    all_context_observations_reach_front_end,
    metric_rows,
)


def test_transformer_output_shape_and_deterministic_initialization() -> None:
    seed_everything(42)
    first = TransformerForecaster(horizon=6)
    seed_everything(42)
    second = TransformerForecaster(horizon=6)
    inputs = torch.zeros((2, 168, 4))
    assert first(inputs).shape == (2, 6)
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())


def test_transformer_front_end_uses_all_168_context_observations() -> None:
    assert all_context_observations_reach_front_end()
    model = TransformerForecaster(horizon=1)
    inputs = torch.randn((1, 168, 4), requires_grad=True)
    model(inputs).sum().backward()
    assert bool((inputs.grad.abs().sum(dim=(0, 2)) > 0).all())


def test_transformer_metrics_remain_in_original_target_units() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["transformer", "transformer"],
            "split": ["test", "test"],
            "horizon": [1, 1],
            "y_true": [0.01, -0.02],
            "y_pred": [0.0, -0.01],
        }
    )
    row = metric_rows(predictions).iloc[0]
    assert row["n"] == 2
    assert np.isclose(row["mae"], 0.01)
    assert np.isclose(row["rmse"], np.sqrt((0.01**2 + 0.01**2) / 2))


def test_saved_transformer_predictions_match_authoritative_index() -> None:
    sample_index = pd.read_parquet("data/processed/task06_sample_index.parquet")
    predictions = pd.read_parquet("outputs/predictions/task08b_transformer_predictions.parquet")
    metadata = json.loads(
        Path("outputs/metadata/task08b_model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["features"] == ["log_return", "log_volume", "high_low_range", "rv_24h"]
    assert metadata["target"] == "original unscaled log_return"
    assert len(predictions) == len(sample_index)
    assert predictions["model"].eq("transformer").all()
    for split in sample_index["split"].unique():
        for horizon in sample_index["horizon"].unique():
            expected = sample_index.loc[
                (sample_index["split"] == split) & (sample_index["horizon"] == horizon),
                "target_end_timestamp",
            ].sort_values().reset_index(drop=True)
            actual = predictions.loc[
                (predictions["split"] == split) & (predictions["horizon"] == horizon),
                "timestamp",
            ].sort_values().reset_index(drop=True)
            pd.testing.assert_series_equal(actual, expected, check_names=False)
