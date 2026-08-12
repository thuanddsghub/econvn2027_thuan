"""Compute paired seed-level Task 18 statistical comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

SEEDS = [42, 123, 2024, 3407, 7777]
HORIZONS = [1, 6, 24]
METRICS = ["crps", "median_mae", "median_rmse", "picp", "interval_width"]
LABELS = {
    "crps": "CRPS",
    "median_mae": "MAE",
    "median_rmse": "RMSE",
    "picp": "PICP90",
    "interval_width": "interval_width",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_rows(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    selected = frame.loc[
        (frame["split"] == "test")
        & (frame["target_step"] == frame["horizon"])
        & (frame["nominal_coverage"] == 0.9)
        & (frame["regime"] == regime)
    ].copy()
    expected = {(seed, horizon, model) for seed in SEEDS for horizon in HORIZONS for model in ("standard_diffusion", "va_diff")}
    actual = set(zip(selected["seed"], selected["horizon"], selected["model"]))
    if actual != expected or len(selected) != 30:
        raise ValueError(f"invalid paired rows for {regime}: expected 30 rows")
    return selected


def test_statistics(values: np.ndarray) -> tuple[float, float, str]:
    t_result = ttest_rel(values[:, 1], values[:, 0])
    try:
        w_result = wilcoxon(values[:, 1], values[:, 0], alternative="two-sided", method="exact")
        wilcoxon_method = "exact"
    except ValueError:
        w_result = wilcoxon(values[:, 1], values[:, 0], alternative="two-sided", method="approx")
        wilcoxon_method = "approx"
    return float(t_result.pvalue), float(w_result.pvalue), wilcoxon_method


def summarize(frame: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected = paired_rows(frame, regime)
    rows: list[dict[str, object]] = []
    paired_records: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for metric in METRICS:
            subset = selected.loc[selected["horizon"] == horizon]
            pivot = subset.pivot(index="seed", columns="model", values=metric).reindex(SEEDS)
            standard = pivot["standard_diffusion"].to_numpy(float)
            va = pivot["va_diff"].to_numpy(float)
            delta = va - standard
            if metric == "picp":
                standard_loss = np.abs(standard - 0.9)
                va_loss = np.abs(va - 0.9)
                percentage = 100.0 * (standard_loss - va_loss) / np.where(standard_loss == 0, np.nan, standard_loss)
                wins = int(np.sum(va_loss < standard_loss))
                direction = "closer_to_0.90_is_better"
            else:
                percentage = 100.0 * (standard - va) / standard
                wins = int(np.sum(va < standard))
                direction = "lower_is_better"
            rows.append(
                {
                    "regime": regime,
                    "horizon": horizon,
                    "metric": LABELS[metric],
                    "metric_column": metric,
                    "mean_standard": float(np.mean(standard)),
                    "mean_va_diff": float(np.mean(va)),
                    "mean_delta_va_minus_standard": float(np.mean(delta)),
                    "std_delta": float(np.std(delta, ddof=1)),
                    "median_delta": float(np.median(delta)),
                    "mean_percentage_improvement": float(np.nanmean(percentage)),
                    "va_wins": wins,
                    "n_seeds": len(SEEDS),
                    "comparison_direction": direction,
                }
            )
            if metric == "crps":
                p_t, p_w, w_method = test_statistics(np.column_stack([standard, va]))
                dz = float(np.mean(delta) / np.std(delta, ddof=1))
                paired_records.append(
                    {
                        "regime": regime,
                        "horizon": horizon,
                        "standard_crps_by_seed": standard.tolist(),
                        "va_diff_crps_by_seed": va.tolist(),
                        "delta_by_seed_va_minus_standard": delta.tolist(),
                        "paired_t_pvalue": p_t,
                        "wilcoxon_pvalue": p_w,
                        "wilcoxon_method": w_method,
                        "cohens_dz": dz,
                        "n_seeds": len(SEEDS),
                        "statistical_power_note": "n=5 paired seeds; p-values are exploratory and not sufficient alone for a significance claim",
                    }
                )
    return pd.DataFrame(rows), paired_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=Path("outputs/final"))
    args = parser.parse_args()
    overall_path = args.final_dir / "final_metrics.csv"
    regime_path = args.final_dir / "final_regime_metrics.csv"
    manifest_path = args.final_dir / "final_manifest.json"
    output_paths = [
        args.final_dir / "statistical_summary.csv",
        args.final_dir / "statistical_regime_summary.csv",
        args.final_dir / "statistical_report.json",
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite immutable Task 18 statistical artifacts")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS" or manifest["validation"]["rv_168h_used"]:
        raise ValueError("final manifest is not an eligible Task 18 input")
    overall = pd.read_csv(overall_path)
    regime = pd.read_csv(regime_path)
    overall_summary, overall_tests = summarize(overall, "Overall")
    regime_summaries: list[pd.DataFrame] = []
    regime_tests: list[dict[str, object]] = []
    for label in ("Low", "Medium", "High"):
        summary, tests = summarize(regime, label)
        regime_summaries.append(summary)
        regime_tests.extend(tests)
    regime_summary = pd.concat(regime_summaries, ignore_index=True)
    overall_summary.to_csv(output_paths[0], index=False)
    regime_summary.to_csv(output_paths[1], index=False)
    report = {
        "task": "18",
        "status": "PASS",
        "inputs": {
            "final_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "final_metrics": {"path": str(overall_path), "sha256": sha256(overall_path)},
            "final_regime_metrics": {"path": str(regime_path), "sha256": sha256(regime_path)},
        },
        "paired_unit": "seed; forecast timestamps are not treated as independent observations",
        "seeds": SEEDS,
        "horizons": HORIZONS,
        "metrics": METRICS,
        "crps_tests_overall": overall_tests,
        "crps_tests_by_regime": regime_tests,
        "interpretation_policy": "n=5 has low statistical power; exact p-values are reported but no significance claim is made from them alone",
        "pre_correction_outputs_used": False,
        "rv_168h_used": False,
        "deterministic": True,
        "output_row_counts": {"overall": len(overall_summary), "regime": len(regime_summary)},
    }
    output_paths[2].write_text(json.dumps(report, indent=2) + "\n")
    print(overall_summary.to_string(index=False))
    print(pd.DataFrame(overall_tests).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
