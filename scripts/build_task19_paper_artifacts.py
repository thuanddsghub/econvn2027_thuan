"""Build deterministic manuscript tables and figures from frozen outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/task19-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINAL = Path("outputs/final")
PAPER = FINAL / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
SEEDS = [42, 123, 2024, 3407, 7777]
HORIZONS = [1, 6, 24]
MODELS = ["standard_diffusion", "va_diff"]
MODEL_LABELS = {"standard_diffusion": "Standard Diffusion", "va_diff": "VA-Diff"}
COLORS = {"standard_diffusion": "#35608D", "va_diff": "#C44E52"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    latex = frame.to_latex(index=False, escape=True, caption=caption, label=label)
    path.write_text(latex)


def setup_axes() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> int:
    if PAPER.exists() and any(PAPER.rglob("*")):
        raise FileExistsError("refusing to overwrite existing Task 19 paper artifacts")
    TABLES.mkdir(parents=True, exist_ok=False)
    FIGURES.mkdir(parents=True, exist_ok=False)
    final_metrics_path = FINAL / "final_metrics.csv"
    final_regime_path = FINAL / "final_regime_metrics.csv"
    statistical_summary_path = FINAL / "statistical_summary.csv"
    statistical_regime_path = FINAL / "statistical_regime_summary.csv"
    statistical_report_path = FINAL / "statistical_report.json"
    manifest_path = FINAL / "final_manifest.json"
    metrics = pd.read_csv(final_metrics_path)
    regime_metrics = pd.read_csv(final_regime_path)
    summary = pd.read_csv(statistical_summary_path)
    regime_summary = pd.read_csv(statistical_regime_path)
    report = json.loads(statistical_report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS" or report["pre_correction_outputs_used"]:
        raise ValueError("invalid or non-final Task 17/18 inputs")
    if sorted(metrics["seed"].unique().tolist()) != SEEDS:
        raise ValueError("frozen metric seed list mismatch")
    if len(metrics) != 30 or len(regime_metrics) != 90:
        raise ValueError("frozen metric dimensions mismatch")

    # Table 1: model means and seed standard deviations for the final test rows.
    rows = []
    selected = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["target_step"] == metrics["horizon"])
        & (metrics["nominal_coverage"] == 0.9)
    ]
    for horizon in HORIZONS:
        for model in MODELS:
            part = selected[(selected["horizon"] == horizon) & (selected["model"] == model)]
            row = {"Horizon": horizon, "Model": MODEL_LABELS[model]}
            for source, label in (
                ("crps", "CRPS"),
                ("median_mae", "MAE"),
                ("median_rmse", "RMSE"),
                ("picp", "PICP90"),
                ("interval_width", "Interval width"),
            ):
                row[label] = f"{part[source].mean():.6f} $\\pm$ {part[source].std(ddof=1):.6f}"
            rows.append(row)
    table1 = pd.DataFrame(rows)
    table1.to_csv(TABLES / "table1_overall_performance.csv", index=False)
    write_latex(table1, TABLES / "table1_overall_performance.tex", "Overall test performance across five paired seeds.", "tab:overall-performance")

    # Table 2: regime CRPS means and seed standard deviations.
    rows = []
    for regime in ("Low", "Medium", "High"):
        for horizon in HORIZONS:
            row = {"Regime": regime, "Horizon": horizon}
            for model in MODELS:
                part = regime_metrics[
                    (regime_metrics["regime"] == regime)
                    & (regime_metrics["horizon"] == horizon)
                    & (regime_metrics["model"] == model)
                    & (regime_metrics["split"] == "test")
                    & (regime_metrics["target_step"] == regime_metrics["horizon"])
                ]
                row[MODEL_LABELS[model]] = f"{part.crps.mean():.6f} $\\pm$ {part.crps.std(ddof=1):.6f}"
            rows.append(row)
    table2 = pd.DataFrame(rows)
    table2.to_csv(TABLES / "table2_regime_crps.csv", index=False)
    write_latex(table2, TABLES / "table2_regime_crps.tex", "Test CRPS by frozen train-only volatility regime.", "tab:regime-crps")

    # Table 3: the statistical comparisons already computed in Task 18.
    tests = pd.DataFrame(report["crps_tests_overall"])
    stat_crps = summary[summary["metric"] == "CRPS"]
    table3 = tests.merge(
        stat_crps[["horizon", "mean_percentage_improvement", "va_wins"]], on="horizon", validate="one_to_one"
    )[
        ["horizon", "delta_by_seed_va_minus_standard", "mean_percentage_improvement", "va_wins", "paired_t_pvalue", "wilcoxon_pvalue", "cohens_dz"]
    ].copy()
    table3["Horizon"] = table3.pop("horizon")
    table3["Mean paired ΔCRPS"] = table3.pop("delta_by_seed_va_minus_standard").map(lambda x: f"{np.mean(x):.6e}" if isinstance(x, list) else "")
    # The report stores per-seed deltas as lists; recover them deterministically.
    table3["Mean paired ΔCRPS"] = [f"{np.mean(x):.6e}" for x in tests["delta_by_seed_va_minus_standard"]]
    table3["% improvement"] = table3.pop("mean_percentage_improvement").map(lambda x: f"{x:.3f}%")
    table3["VA wins / 5"] = table3.pop("va_wins").map(lambda x: f"{int(x)} / 5")
    table3["Paired t p"] = table3.pop("paired_t_pvalue").map(lambda x: f"{x:.6f}")
    table3["Wilcoxon p"] = table3.pop("wilcoxon_pvalue").map(lambda x: f"{x:.6f}")
    table3["Cohen's dz"] = table3.pop("cohens_dz").map(lambda x: f"{x:.3f}")
    table3 = table3[["Horizon", "Mean paired ΔCRPS", "% improvement", "VA wins / 5", "Paired t p", "Wilcoxon p", "Cohen's dz"]]
    table3.to_csv(TABLES / "table3_statistical_robustness.csv", index=False)
    write_latex(table3, TABLES / "table3_statistical_robustness.tex", "Paired seed-level CRPS robustness statistics.", "tab:statistical-robustness")

    setup_axes()
    # Figure 1: the frozen final snapshot contains nominal 90% coverage only.
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        for model in MODELS:
            part = selected[(selected["horizon"] == horizon) & (selected["model"] == model)]
            axis.scatter([0.9], [part.picp.mean()], s=45, color=COLORS[model], label=MODEL_LABELS[model])
        axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axis.set_title(f"H = {horizon}")
        axis.set_xlim(0.84, 0.96)
        axis.set_ylim(0.84, 0.96)
        axis.set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("90% predictive interval calibration")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_calibration.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "figure1_calibration.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: regime CRPS means from Task 18.
    crps_regime = regime_summary[regime_summary["metric"] == "CRPS"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
    x = np.arange(3)
    for axis, horizon in zip(axes, HORIZONS):
        part = crps_regime[crps_regime["horizon"] == horizon]
        for offset, model in zip((-0.16, 0.16), MODELS):
            column = "mean_standard" if model == "standard_diffusion" else "mean_va_diff"
            values = [
                part.loc[(part["regime"] == regime) & (part["metric_column"] == "crps"), column].iloc[0]
                for regime in ("Low", "Medium", "High")
            ]
            axis.bar(x + offset, values, width=0.3, color=COLORS[model], label=MODEL_LABELS[model])
        axis.set_xticks(x, ["Low", "Medium", "High"])
        axis.set_title(f"H = {horizon}")
        axis.set_xlabel("Volatility regime")
    axes[0].set_ylabel("CRPS (lower is better)")
    axes[-1].legend(frameon=False)
    fig.suptitle("CRPS by frozen volatility regime")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_regime_crps.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "figure2_regime_crps.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: exact paired per-seed deltas from Task 18 report.
    paired = pd.DataFrame(report["crps_tests_overall"])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        row = paired[paired["horizon"] == horizon].iloc[0]
        deltas = np.asarray(row["delta_by_seed_va_minus_standard"])
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.scatter(range(1, 6), deltas, color=COLORS["va_diff"], s=38)
        axis.set_title(f"H = {horizon}")
        axis.set_xticks(range(1, 6), [str(seed) for seed in SEEDS], rotation=45)
        axis.set_xlabel("Seed")
    axes[0].set_ylabel("ΔCRPS (VA − Standard)")
    fig.suptitle("Paired five-seed CRPS differences")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_paired_crps.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "figure3_paired_crps.png", bbox_inches="tight")
    plt.close(fig)

    captions = {
        "figure1": "Nominal 90% coverage and empirical coverage across horizons; points are five-seed means.",
        "figure2": "Mean test CRPS by frozen train-only volatility regime across five paired seeds.",
        "figure3": "Per-seed VA-Diff minus Standard Diffusion CRPS; negative values favor VA-Diff.",
    }
    (PAPER / "captions.json").write_text(json.dumps(captions, indent=2) + "\n")
    validation = {
        "status": "PASS",
        "source_sha256": {str(path): sha256(path) for path in (final_metrics_path, final_regime_path, statistical_summary_path, statistical_regime_path, statistical_report_path, manifest_path)},
        "pre_correction_outputs_used": False,
        "metric_recomputation_from_samples": False,
        "deterministic": True,
        "table_dimensions": {"table1": [len(table1), len(table1.columns)], "table2": [len(table2), len(table2.columns)], "table3": [len(table3), len(table3.columns)]},
        "figure_data_source": "frozen Task 17/18 aggregates only",
        "calibration_levels_available": [0.9],
    }
    (PAPER / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
