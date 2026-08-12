"""Regression tests for paired Task 18 statistical comparisons."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.compute_task18_statistics import paired_rows, summarize


def test_task18_has_five_paired_seeds_and_expected_outputs():
    overall = pd.read_csv("outputs/final/final_metrics.csv")
    summary, tests = summarize(overall, "Overall")
    assert len(summary) == 15
    assert len(tests) == 3
    assert set(summary["n_seeds"]) == {5}
    assert all(len(row["delta_by_seed_va_minus_standard"]) == 5 for row in tests)


def test_task18_rejects_unpaired_rows():
    overall = pd.read_csv("outputs/final/final_metrics.csv")
    broken = overall.iloc[:-1]
    with pytest.raises(ValueError, match="invalid paired rows"):
        paired_rows(broken, "Overall")


def test_task18_outputs_are_deterministic_and_complete():
    report = json.loads(Path("outputs/final/statistical_report.json").read_text())
    assert report["status"] == "PASS"
    assert report["pre_correction_outputs_used"] is False
    assert report["rv_168h_used"] is False
    assert report["output_row_counts"] == {"overall": 15, "regime": 45}
