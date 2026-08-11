import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fit_task08a_lstm import LSTMForecaster, metric_rows, seed_everything


def test_lstm_output_shape_and_deterministic_initialization() -> None:
    seed_everything(42)
    first = LSTMForecaster(horizon=6)
    seed_everything(42)
    second = LSTMForecaster(horizon=6)
    inputs = torch.zeros((2, 168, 4))
    assert first(inputs).shape == (2, 6)
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())


def test_task08a_metrics_are_computed_in_original_target_units() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["lstm", "lstm"],
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


def test_saved_lstm_predictions_match_authoritative_task06_index() -> None:
    sample_index = pd.read_parquet("data/processed/task06_sample_index.parquet")
    predictions = pd.read_parquet("outputs/predictions/task08a_lstm_predictions.parquet")
    metadata = json.loads(
        Path("outputs/metadata/task08a_model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["features"] == ["log_return", "log_volume", "high_low_range", "rv_24h"]
    assert metadata["target"] == "original unscaled log_return"
    assert len(predictions) == len(sample_index)
    assert predictions["model"].eq("lstm").all()
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
