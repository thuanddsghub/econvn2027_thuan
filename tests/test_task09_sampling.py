import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_task09_sampling_artifacts_are_compact_and_complete() -> None:
    report = json.loads(
        Path("outputs/metadata/task09_sampling_validation_report.json").read_text()
    )
    metrics = pd.read_csv("outputs/metrics/task09_sampling_metrics.csv")
    assert report["status"] == "PASS"
    assert report["final_sample_count"] == 100
    assert report["deterministic_sampling"]
    assert set(metrics["sample_count"]) == {10, 25, 50, 100}
    archive = np.load("outputs/predictions/task09_samples/test_h24.npz")
    assert archive["samples"].shape[1:] == (100, 24)
    assert np.isfinite(archive["samples"]).all()


def test_task09_sampling_metrics_have_monotonic_sample_count_grid() -> None:
    metrics = pd.read_csv("outputs/metrics/task09_sampling_metrics.csv")
    for _, group in metrics.groupby(["split", "horizon", "target_step"]):
        assert group["sample_count"].tolist() == [10, 25, 50, 100]
