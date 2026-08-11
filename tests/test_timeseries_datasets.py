from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import torch

from va_diff.data.timeseries import (
    MODEL_FEATURES,
    ETHUSDTimeseriesDataset,
    build_sample_index,
    make_dataloader,
)


def make_frame(n: int = 200, gap_at: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    if gap_at is not None:
        timestamps = [start + timedelta(hours=i) for i in range(n + 1)]
        timestamps.pop(gap_at)
    features = pd.DataFrame({"timestamp": timestamps})
    for column, value in zip(MODEL_FEATURES, [1.0, 2.0, 3.0, 4.0]):
        features[column] = value
    features["log_return"] = np.arange(n, dtype=float)
    split_index = pd.DataFrame({"timestamp": timestamps, "split": "train"})
    return features, split_index


def test_first_middle_last_valid_samples_and_shapes() -> None:
    features, split_index = make_frame()
    index, rejected = build_sample_index(features, split_index, horizons=(1, 6, 24))
    dataset = ETHUSDTimeseriesDataset(features, index, split="train", horizon=6)

    assert len(dataset) == 27
    x_first, y_first = dataset[0]
    x_middle, y_middle = dataset[len(dataset) // 2]
    x_last, y_last = dataset[-1]
    assert x_first.shape == (168, 4) and y_first.shape == (6,)
    assert x_middle.shape == (168, 4) and y_middle.shape == (6,)
    assert x_last.shape == (168, 4) and y_last.shape == (6,)
    assert dataset.sample_index.iloc[0]["endpoint_timestamp"] == pd.Timestamp("2023-01-07 23:00", tz="UTC")
    assert dataset.sample_index.iloc[-1]["endpoint_timestamp"] == pd.Timestamp("2023-01-09 01:00", tz="UTC")
    assert rejected["train/horizon_6"]["insufficient_context"] == 167
    assert rejected["train/horizon_6"]["insufficient_target"] == 5


def test_gap_rejects_samples_and_loader_is_reproducible() -> None:
    features, split_index = make_frame(gap_at=180)
    index, rejected = build_sample_index(features, split_index, horizons=(1,))
    assert rejected["train/horizon_1"]["context_gap"] > 0
    assert rejected["train/horizon_1"]["target_gap"] > 0
    dataset = ETHUSDTimeseriesDataset(features, index, split="train", horizon=1)
    loader = make_dataloader(dataset, batch_size=4)
    first_x, first_y = next(iter(loader))
    assert first_x.shape == (4, 168, 4)
    assert first_y.shape == (4, 1)
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_validation_target_can_use_prior_split_context() -> None:
    features, split_index = make_frame(n=220)
    boundary = features.loc[190, "timestamp"]
    split_index.loc[split_index["timestamp"] >= boundary, "split"] = "validation"

    index, _ = build_sample_index(features, split_index, horizons=(1,))
    validation = index[index["split"] == "validation"]

    assert validation.iloc[0]["endpoint_row"] == 189
    assert validation.iloc[0]["endpoint_timestamp"] < boundary
    assert validation.iloc[0]["target_end_timestamp"] == boundary
