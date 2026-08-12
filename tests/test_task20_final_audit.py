"""Regression checks for the final experiment audit."""

import json
from pathlib import Path


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
