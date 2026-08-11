import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fit_task07b_baselines import metric_rows, zero_forecast


def test_zero_return_forecast_is_exactly_zero() -> None:
    assert np.array_equal(zero_forecast(123.0, 24), np.zeros(24))


def test_metrics_recompute_from_prediction_rows() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["zero_return", "zero_return"],
            "split": ["test", "test"],
            "horizon": [1, 1],
            "y_true": [1.0, -2.0],
            "y_pred": [0.0, 0.0],
        }
    )
    metrics = metric_rows(predictions).iloc[0]
    assert metrics["mae"] == 1.5
    assert np.isclose(metrics["rmse"], np.sqrt(2.5))


def test_saved_predictions_match_authoritative_task06_sample_index() -> None:
    sample_path = Path("data/processed/task06_sample_index.parquet")
    manifest_path = Path("data/metadata/task06_dataset_manifest.json")
    prediction_path = Path("outputs/predictions/task07b_predictions.parquet")
    sample_index = pd.read_parquet(sample_path)
    predictions = pd.read_parquet(prediction_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()

    assert manifest["sample_index_sha256"] == actual_checksum
    assert len(predictions) == len(sample_index) * predictions["model"].nunique()
    for model in predictions["model"].unique():
        for split in sample_index["split"].unique():
            for horizon in sample_index["horizon"].unique():
                expected = sample_index.loc[
                    (sample_index["split"] == split) & (sample_index["horizon"] == horizon),
                    "target_end_timestamp",
                ].sort_values().reset_index(drop=True)
                actual = predictions.loc[
                    (predictions["model"] == model)
                    & (predictions["split"] == split)
                    & (predictions["horizon"] == horizon),
                    "timestamp",
                ].sort_values().reset_index(drop=True)
                pd.testing.assert_series_equal(actual, expected, check_names=False)
