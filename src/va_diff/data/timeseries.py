"""Leakage-safe PyTorch time-series datasets for Task 06."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

MODEL_FEATURES = ("log_return", "log_volume", "high_low_range", "rv_24h")
TARGET_COLUMN = "log_return"
EXPECTED_INTERVAL = pd.Timedelta(hours=1)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SampleSpec:
    context_length: int = 168


def _rejection_reason(
    timestamp_ns: np.ndarray,
    split_values: np.ndarray,
    model_values: np.ndarray,
    target_values: np.ndarray,
    endpoint: int,
    split_name: str,
    horizon: int,
    context_length: int,
) -> str | None:
    context_start = endpoint - context_length + 1
    target_end = endpoint + horizon
    if context_start < 0:
        return "insufficient_context"
    if target_end >= len(timestamp_ns):
        return "insufficient_target"
    span_splits = split_values[context_start : target_end + 1]
    if not np.all(span_splits == split_name):
        return "split_boundary"
    span_timestamps = timestamp_ns[context_start : target_end + 1]
    gap_positions = np.diff(span_timestamps) != EXPECTED_INTERVAL.value
    if gap_positions[: context_length - 1].any():
        return "context_gap"
    if gap_positions[context_length - 1 :].any():
        return "target_gap"
    if np.isnan(model_values[context_start : endpoint + 1]).any():
        return "nan_context"
    if np.isnan(target_values[endpoint + 1 : target_end + 1]).any():
        return "nan_target"
    return None


def build_sample_index(
    features: pd.DataFrame,
    split_index: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 6, 24),
    context_length: int = 168,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Build valid endpoint rows and mutually exclusive rejection counts."""
    required_features = {"timestamp", *MODEL_FEATURES}
    missing_features = required_features.difference(features.columns)
    if missing_features:
        raise ValueError(f"missing feature columns: {sorted(missing_features)}")
    if set(split_index.columns) < {"timestamp", "split"}:
        raise ValueError("split index must contain timestamp and split columns")
    if context_length < 1 or any(horizon < 1 for horizon in horizons):
        raise ValueError("context_length and horizons must be positive")

    data = features.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    labels = split_index.copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    if not data["timestamp"].is_monotonic_increasing or data["timestamp"].duplicated().any():
        raise ValueError("feature timestamps must be strictly increasing and unique")
    if not labels["timestamp"].is_unique or set(labels["timestamp"]) != set(data["timestamp"]):
        raise ValueError("split index timestamps must match feature timestamps exactly")
    labels = labels.set_index("timestamp").loc[data["timestamp"]].reset_index()
    if labels["split"].isna().any() or not labels["split"].isin(SPLITS).all():
        raise ValueError("split index contains missing or unknown split labels")

    timestamp_ns = data["timestamp"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    split_values = labels["split"].to_numpy()
    model_values = data[list(MODEL_FEATURES)].to_numpy(dtype=float)
    target_values = data[TARGET_COLUMN].to_numpy(dtype=float)
    timestamps = data["timestamp"].to_numpy()
    gap = np.concatenate(([False], np.diff(timestamp_ns) != EXPECTED_INTERVAL.value))
    gap_prefix = np.concatenate(([0], np.cumsum(gap, dtype=np.int64)))
    context_nan = np.isnan(model_values).any(axis=1)
    target_nan = np.isnan(target_values)
    context_nan_prefix = np.concatenate(([0], np.cumsum(context_nan, dtype=np.int64)))
    target_nan_prefix = np.concatenate(([0], np.cumsum(target_nan, dtype=np.int64)))
    records: list[pd.DataFrame] = []
    rejection_counts: dict[str, dict[str, int]] = {}
    for split_name in SPLITS:
        # Candidate endpoints are the rows immediately before each requested-split
        # target start. This permits historical context from the prior split.
        target_starts = np.flatnonzero(split_values[1:] == split_name) + 1
        endpoints = target_starts - 1
        for horizon in horizons:
            key = f"{split_name}/horizon_{horizon}"
            counts: Counter[str] = Counter()
            insufficient_context = endpoints < context_length - 1
            insufficient_target = (~insufficient_context) & (endpoints + horizon >= len(data))
            counts["insufficient_context"] = int(insufficient_context.sum())
            counts["insufficient_target"] = int(insufficient_target.sum())
            eligible = ~(insufficient_context | insufficient_target)
            valid_endpoints = endpoints[eligible]
            starts = valid_endpoints - context_length + 1
            ends = valid_endpoints + horizon
            split_prefix = np.concatenate(([0], np.cumsum(split_values == split_name, dtype=np.int64)))
            same_split = (
                split_prefix[ends + 1] - split_prefix[valid_endpoints + 1] == ends - valid_endpoints
            )
            counts["split_boundary"] = int((~same_split).sum())
            valid_endpoints = valid_endpoints[same_split]
            starts = starts[same_split]
            ends = ends[same_split]
            context_gap = (gap_prefix[valid_endpoints + 1] - gap_prefix[starts + 1]) > 0
            target_gap = (gap_prefix[ends + 1] - gap_prefix[valid_endpoints + 1]) > 0
            counts["context_gap"] = int(context_gap.sum())
            target_gap_only = target_gap & ~context_gap
            counts["target_gap"] = int(target_gap_only.sum())
            keep = ~(context_gap | target_gap)
            valid_endpoints = valid_endpoints[keep]
            starts = starts[keep]
            ends = ends[keep]
            context_has_nan = (
                context_nan_prefix[valid_endpoints + 1] - context_nan_prefix[starts] > 0
            )
            target_has_nan = target_nan_prefix[ends + 1] - target_nan_prefix[valid_endpoints + 1] > 0
            counts["nan_context"] = int(context_has_nan.sum())
            counts["nan_target"] = int((target_has_nan & ~context_has_nan).sum())
            keep = ~(context_has_nan | target_has_nan)
            valid_endpoints = valid_endpoints[keep]
            records.append(
                pd.DataFrame(
                    {
                        "split": split_name,
                        "horizon": horizon,
                        "endpoint_row": valid_endpoints.astype(np.int64),
                        "endpoint_timestamp": timestamps[valid_endpoints],
                        "target_start_timestamp": timestamps[valid_endpoints + 1],
                        "context_start_timestamp": timestamps[valid_endpoints - context_length + 1],
                        "target_end_timestamp": timestamps[valid_endpoints + horizon],
                    }
                )
            )
            rejection_counts[key] = {
                reason: count for reason, count in sorted(counts.items()) if count
            }
    sample_index = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if not sample_index.empty:
        for column in [
            "endpoint_timestamp",
            "target_start_timestamp",
            "context_start_timestamp",
            "target_end_timestamp",
        ]:
            sample_index[column] = pd.to_datetime(sample_index[column], utc=True)
    return sample_index, rejection_counts


class ETHUSDTimeseriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset returning context features and future log-return targets."""

    def __init__(
        self,
        features: pd.DataFrame,
        sample_index: pd.DataFrame,
        *,
        split: str,
        horizon: int,
        context_length: int = 168,
    ) -> None:
        self.features = features.reset_index(drop=True)
        self.sample_index = sample_index.loc[
            (sample_index["split"] == split) & (sample_index["horizon"] == horizon)
        ].reset_index(drop=True)
        self.horizon = horizon
        self.context_length = context_length
    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        endpoint = int(self.sample_index.iloc[index]["endpoint_row"])
        context = self.features.iloc[endpoint - self.context_length + 1 : endpoint + 1][
            list(MODEL_FEATURES)
        ].to_numpy(dtype=np.float32, copy=True)
        target = self.features.iloc[endpoint + 1 : endpoint + self.horizon + 1][
            TARGET_COLUMN
        ].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(context), torch.from_numpy(target)


def make_dataloader(
    dataset: ETHUSDTimeseriesDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a deterministic DataLoader; chronological order is the default."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
