"""Validation tests for the immutable Task 17 manuscript snapshot."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.freeze_task17 import write_once


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_task17_manifest_hashes_and_counts_resolve():
    root = Path("outputs/final")
    manifest = json.loads((root / "final_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["final_metric_row_counts"] == {"overall": 30, "regime": 90}
    for item in manifest["hashes"].values():
        path = Path(item["path"])
        assert path.exists()
        assert _sha256(path) == item["sha256"]
    assert manifest["frozen_regime_thresholds"]["q33"] == pytest.approx(0.027575830399575955)
    assert manifest["frozen_regime_thresholds"]["q67"] == pytest.approx(0.04549905819580233)
    assert manifest["validation"]["corrected_ddpm_posterior_variance"]
    assert not manifest["validation"]["rv_168h_used"]
    sample_index = pd.read_parquet("data/processed/task06_sample_index.parquet")
    actual_counts = {
        f"{split}/horizon_{horizon}": int(
            ((sample_index["split"] == split) & (sample_index["horizon"] == horizon)).sum()
        )
        for split in ("train", "validation", "test")
        for horizon in (1, 6, 24)
    }
    assert manifest["task06_sample_counts"] == actual_counts

    features = pd.read_parquet("data/processed/eth_usd_hourly_features.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    train = features.loc[
        features["timestamp"] < pd.Timestamp("2024-01-01T00:00:00Z"), "log_return"
    ].dropna()
    target_scaler = json.loads((root / "target_scaler.json").read_text())
    assert target_scaler["mean"] == pytest.approx(train.mean())
    assert target_scaler["std_ddof_1"] == pytest.approx(train.std(ddof=1))


def test_task17_metrics_are_exact_copies_of_task16_sources():
    assert _sha256(Path("outputs/final/final_metrics.csv")) == _sha256(
        Path("outputs/metrics/task16_multiseed_metrics.csv")
    )
    assert _sha256(Path("outputs/final/final_regime_metrics.csv")) == _sha256(
        Path("outputs/metrics/task16_regime_metrics.csv")
    )
    assert len(pd.read_csv("outputs/final/final_metrics.csv")) == 30
    assert len(pd.read_csv("outputs/final/final_regime_metrics.csv")) == 90


def test_task17_freeze_writes_once(tmp_path):
    path = tmp_path / "immutable.json"
    write_once(path, "first\n")
    with pytest.raises(FileExistsError):
        write_once(path, "second\n")
    assert path.read_text() == "first\n"
