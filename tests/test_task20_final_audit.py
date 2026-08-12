"""Regression checks for the final experiment audit."""

import json
from pathlib import Path

import pandas as pd

from scripts.audit_task20_final_experiment import derive_conclusions, expected_paper_tables


def test_task20_audit_passes_without_inconsistencies():
    report = json.loads(Path("outputs/final/final_audit_report.json").read_text())
    assert report["status"] == "PASS"
    assert report["inconsistencies_found"] == []
    assert all(item["status"] == "PASS" for item in report["checks"])


def test_task20_report_contains_required_sections():
    report = Path("outputs/final/FINAL_EXPERIMENT_REPORT.md").read_text()
    for section in (
        "Dataset and date range",
        "Split and preprocessing",
        "Models and diffusion correction",
        "Final multi-seed metrics",
        "Regime results",
        "Calibration summary",
        "Statistical robustness",
        "Known limitations",
        "Manuscript-safe conclusions",
        "Artifact provenance and checksums",
    ):
        assert f"## {section}" in report


def test_task20_reconstructs_paper_tables_from_frozen_sources():
    root = Path("outputs/final")
    expected = expected_paper_tables(
        pd.read_csv(root / "final_metrics.csv"),
        pd.read_csv(root / "final_regime_metrics.csv"),
        pd.read_csv(root / "statistical_summary.csv"),
        json.loads((root / "statistical_report.json").read_text()),
    )
    actual = (
        pd.read_csv(root / "paper/tables/table1_overall_performance.csv"),
        pd.read_csv(root / "paper/tables/table2_regime_crps.csv"),
        pd.read_csv(root / "paper/tables/table3_statistical_robustness.csv"),
    )
    assert all(actual_table.equals(expected_table) for actual_table, expected_table in zip(actual, expected))


def test_task20_conclusions_are_derived_not_hardcoded():
    root = Path("outputs/final")
    summary = pd.read_csv(root / "statistical_summary.csv")
    regimes = pd.read_csv(root / "statistical_regime_summary.csv")
    report = json.loads((root / "statistical_report.json").read_text())
    metrics = pd.read_csv(root / "final_metrics.csv")
    calibration = metrics.groupby(["model", "horizon"], as_index=False)["picp"].mean()

    conclusions = derive_conclusions(summary, regimes, report, calibration)
    assert conclusions["small_h1_h6_improvements"]

    changed = summary.copy()
    changed.loc[(changed["metric"] == "CRPS") & (changed["horizon"] == 1), "mean_percentage_improvement"] = -0.1
    changed_conclusions = derive_conclusions(changed, regimes, report, calibration)
    assert not changed_conclusions["small_h1_h6_improvements"]
