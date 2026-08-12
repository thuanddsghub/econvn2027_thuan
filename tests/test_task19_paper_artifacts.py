"""Validation tests for Task 19 manuscript artifacts."""

import json
from pathlib import Path

import pandas as pd


def test_task19_tables_have_expected_dimensions_and_sources():
    root = Path("outputs/final/paper")
    report = json.loads((root / "validation_report.json").read_text())
    assert report["status"] == "PASS"
    assert report["pre_correction_outputs_used"] is False
    assert report["metric_recomputation_from_samples"] is False
    assert report["table_dimensions"] == {
        "table1": [6, 7],
        "table2": [9, 4],
        "table3": [3, 7],
    }
    assert len(pd.read_csv(root / "tables/table1_overall_performance.csv")) == 6
    assert len(pd.read_csv(root / "tables/table2_regime_crps.csv")) == 9
    assert len(pd.read_csv(root / "tables/table3_statistical_robustness.csv")) == 3


def test_task19_figures_exported_in_both_formats():
    figures = Path("outputs/final/paper/figures")
    for stem in ("figure1_calibration", "figure2_regime_crps", "figure3_paired_crps"):
        for suffix in (".pdf", ".png"):
            artifact = figures / f"{stem}{suffix}"
            assert artifact.exists()
            assert artifact.stat().st_size > 0
