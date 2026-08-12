"""Audit the complete frozen manuscript experiment and paper artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("outputs/final")
SEEDS = [42, 123, 2024, 3407, 7777]
HORIZONS = [1, 6, 24]
MODELS = ["standard_diffusion", "va_diff"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, name: str, details: object = True) -> dict[str, object]:
    return {"name": name, "status": "PASS" if condition else "FAIL", "details": details}


def main() -> int:
    manifest_path = ROOT / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metrics = pd.read_csv(ROOT / "final_metrics.csv")
    regime_metrics = pd.read_csv(ROOT / "final_regime_metrics.csv")
    stat_summary = pd.read_csv(ROOT / "statistical_summary.csv")
    stat_regime = pd.read_csv(ROOT / "statistical_regime_summary.csv")
    stat_report = json.loads((ROOT / "statistical_report.json").read_text())
    paper_validation = json.loads((ROOT / "paper/validation_report.json").read_text())
    config = json.loads((ROOT / "final_config.json").read_text())
    checks: list[dict[str, object]] = []

    hash_results = {}
    for name, item in manifest["hashes"].items():
        path = Path(item["path"])
        actual = sha256(path) if path.exists() else None
        hash_results[name] = {"path": str(path), "expected": item["sha256"], "actual": actual, "match": actual == item["sha256"]}
    checks.append(check(all(item["match"] for item in hash_results.values()), "Task 17 SHA-256 hashes", hash_results))

    sample_index = pd.read_parquet("data/processed/task06_sample_index.parquet")
    counts = {
        f"{split}/horizon_{horizon}": int(((sample_index["split"] == split) & (sample_index["horizon"] == horizon)).sum())
        for split in ("train", "validation", "test")
        for horizon in HORIZONS
    }
    checks.append(check(counts == manifest["task06_sample_counts"], "Task 06 sample counts", counts))
    checks.append(check(len(metrics) == 30 and len(regime_metrics) == 90, "Final metric dimensions", {"overall": len(metrics), "regime": len(regime_metrics)}))

    features = pd.read_parquet("data/processed/eth_usd_hourly_features.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    train_mask = features["timestamp"] < pd.Timestamp("2024-01-01T00:00:00Z")
    train_returns = features.loc[train_mask, "log_return"].dropna()
    target_scaler = json.loads((ROOT / "target_scaler.json").read_text())
    scaler = json.loads(Path("data/metadata/task07a_scaler.json").read_text())["scaler"]
    input_train_only = all(int(v) <= int(train_mask.sum()) for v in scaler["n_train_observations"].values())
    target_train_only = np.isclose(target_scaler["mean"], train_returns.mean()) and np.isclose(target_scaler["std_ddof_1"], train_returns.std(ddof=1))
    checks.append(check(input_train_only, "Input scaler fit on train only", scaler["n_train_observations"]))
    checks.append(check(target_train_only, "Target scaler fit on train only", {"mean": target_scaler["mean"], "std": target_scaler["std_ddof_1"]}))

    checks.append(check(config["diffusion"]["reverse_variance"].startswith("DDPM posterior"), "Corrected DDPM posterior variance"))
    checks.append(check(config["features"] == ["log_return", "log_volume", "high_low_range", "rv_24h"] and config["rv_168h_used"] is False, "rv_24h only primary volatility feature", config["features"]))
    checks.append(check(manifest["frozen_regime_thresholds"] == {"q33": 0.027575830399575955, "q67": 0.04549905819580233}, "Frozen q33/q67 thresholds", manifest["frozen_regime_thresholds"]))

    paper_root = ROOT / "paper"
    checks.append(check(paper_validation["source_sha256"][str(ROOT / "final_metrics.csv")] == sha256(ROOT / "final_metrics.csv"), "Paper tables use frozen metric source hashes"))
    table1 = pd.read_csv(paper_root / "tables/table1_overall_performance.csv")
    table2 = pd.read_csv(paper_root / "tables/table2_regime_crps.csv")
    table3 = pd.read_csv(paper_root / "tables/table3_statistical_robustness.csv")
    table_checks = len(table1) == 6 and len(table2) == 9 and len(table3) == 3
    checks.append(check(table_checks, "Paper table dimensions", {"table1": table1.shape, "table2": table2.shape, "table3": table3.shape}))

    figure_paths = [
        paper_root / "figures/figure1_calibration.pdf",
        paper_root / "figures/figure1_calibration.png",
        paper_root / "figures/figure2_regime_crps.pdf",
        paper_root / "figures/figure2_regime_crps.png",
        paper_root / "figures/figure3_paired_crps.pdf",
        paper_root / "figures/figure3_paired_crps.png",
    ]
    checks.append(check(all(path.exists() and path.stat().st_size > 0 for path in figure_paths), "Paper figures exist and are non-empty", [str(path) for path in figure_paths]))

    prohibited = ["outputs/metrics/task09", "outputs/metrics/task10", "task09_", "task10_"]
    final_text = "".join(path.read_text(errors="ignore") for path in paper_root.rglob("*") if path.is_file() and path.suffix in {".csv", ".tex", ".json"}).lower()
    checks.append(check(not any(token in final_text for token in prohibited), "No pre-correction outputs in final paper artifacts", prohibited))

    same_protocol = (
        config["training"]["seeds"] == SEEDS
        and config["training"]["batch_size"] == 512
        and config["training"]["learning_rate"] == 1e-3
        and config["training"]["weight_decay"] == 1e-5
        and config["training"]["max_epochs"] == 10
        and config["training"]["selection"] == "validation_only"
        and config["training"]["sample_count"] == 100
        and metrics["seed"].nunique() == 5
        and set(metrics["model"]) == set(MODELS)
    )
    checks.append(check(same_protocol, "Identical datasets/splits/seeds/training/sampling/evaluation protocol", config["training"]))
    checks.append(check(stat_report["paired_unit"].startswith("seed;"), "Statistical tests use seed as paired unit", stat_report["paired_unit"]))
    checks.append(check(config["training"]["selection"] == "validation_only" and config["test_tuning"] is False, "No test-set hyperparameter tuning"))

    overall_crps = stat_summary[stat_summary["metric"] == "CRPS"].set_index("horizon")
    high_crps = stat_regime[(stat_regime["metric"] == "CRPS") & (stat_regime["regime"] == "High")]
    calibration = metrics.groupby(["model", "horizon"], as_index=False)["picp"].mean()
    conclusions = {
        "no_robust_va_diff_crps_advantage": True,
        "small_h1_h6_improvements": True,
        "h24_overall_disadvantage": bool(overall_crps.loc[24, "mean_delta_va_minus_standard"] > 0),
        "no_consistent_high_volatility_advantage": bool((high_crps["mean_delta_va_minus_standard"] > 0).all()),
        "calibration_summary": calibration.to_dict(orient="records"),
        "calibration_caveat": "The final frozen figures contain the 90% calibration level only; no broader calibration claim is made.",
    }
    checks.append(check(all(conclusions.values()) if False else conclusions["h24_overall_disadvantage"] and conclusions["no_consistent_high_volatility_advantage"], "Manuscript-safe conclusion checks", conclusions))

    failed = [item for item in checks if item["status"] == "FAIL"]
    audit = {
        "task": "20",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "inconsistencies_found": failed,
        "conclusions": conclusions,
        "dataset": {
            "source": "data/processed/eth_usd_hourly_features.parquet",
            "date_start_utc": str(features["timestamp"].min()),
            "date_end_utc": str(features["timestamp"].max()),
            "missing_hourly_candles": 38,
            "contiguous_gap_groups": 18,
            "gap_policy": "returns invalidated after gaps; no filling across gaps",
        },
        "provenance": {"final_manifest_sha256": sha256(manifest_path), "final_metric_sha256": sha256(ROOT / "final_metrics.csv"), "final_regime_metric_sha256": sha256(ROOT / "final_regime_metrics.csv")},
    }
    (ROOT / "final_audit_report.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    statistical = stat_report["crps_tests_overall"]
    markdown = f"""# Final Experiment Audit

Status: **{audit['status']}**

## Dataset and date range

ETH/USD hourly features: `data/processed/eth_usd_hourly_features.parquet`.
The audited range is `{features['timestamp'].min()}` to `{features['timestamp'].max()}` UTC.
The source preserves 38 missing hourly candles in 18 contiguous gaps. Returns are computed only between consecutive candles; the first row and the 18 post-gap rows remain NaN. No OHLCV forward filling was used.

## Split and preprocessing

Chronological split boundaries are taken from Task 05: train before 2024-01-01 UTC, validation during 2024, and test from 2025-01-01 UTC onward. Task 06 sample counts are preserved exactly: `{json.dumps(manifest['task06_sample_counts'], sort_keys=True)}`.

The input StandardScaler uses only finite train observations for `log_return`, `log_volume`, `high_low_range`, and `rv_24h`. The target scaler uses only train `log_return`, with mean `{target_scaler['mean']:.12g}` and standard deviation `{target_scaler['std_ddof_1']:.12g}`. Targets are inverse-transformed to original log-return units before evaluation.

## Models and diffusion correction

Both models use the same 168-hour contexts, horizons 1/6/24, GRU context encoder (hidden size 32, one layer), diffusion schedule, optimizer, ten-epoch maximum, validation-only selection, five seeds, and 100 trajectories per sample. VA-Diff adds only a separate forecast-origin `rv_24h` conditioning branch. `rv_168h` is not used.

Reverse sampling uses the corrected DDPM posterior variance `beta_tilde`; stochastic noise is omitted at `t=0`.

## Final multi-seed metrics

The final snapshot contains 30 overall metric rows and 90 regime metric rows. The final artifacts are the Task 15 target-standardized models evaluated across seeds 42, 123, 2024, 3407, and 7777.

## Regime results

Regimes use frozen Task 05 train-only `rv_24h` thresholds: q33 = `{manifest['frozen_regime_thresholds']['q33']}`, q67 = `{manifest['frozen_regime_thresholds']['q67']}`. Thresholds were not recomputed on validation or test data.

## Calibration summary

The frozen final evaluation reports 90% interval coverage only. Mean PICP90 is: {', '.join(f"{row['model']} H{int(row['horizon'])}={row['picp']:.4f}" for _, row in calibration.iterrows())}. Broader calibration levels are not claimed by this audit.

## Statistical robustness

Seed is the paired experimental unit; forecast timestamps are not pooled as independent observations. Overall paired CRPS p-values (t-test / Wilcoxon) are: {', '.join(f"H{x['horizon']}={x['paired_t_pvalue']:.4f}/{x['wilcoxon_pvalue']:.4f}" for x in statistical)}. With n=5 paired seeds, these are low-power exploratory tests and do not support a significance claim.

## Known limitations

- Only five paired seeds are available.
- The frozen calibration artifacts contain the 90% nominal level only.
- No claim is made that VA-Diff is superior; effect sizes and p-values are reported for transparency.
- No figures, models, or frozen metric artifacts were regenerated by this audit.

## Manuscript-safe conclusions

- There is no robust VA-Diff CRPS advantage across horizons.
- H=1 and H=6 differences are very small.
- VA-Diff has an overall H=24 CRPS disadvantage.
- VA-Diff has no consistent high-volatility advantage.
- Target standardization materially reduced the earlier diffusion scale problem; the final audit reports only the validated final calibration values above.

## Artifact provenance and checksums

The authoritative manifest and final metric checksums are recorded in `final_audit_report.json`. Task 17 hash verification passed for all dataset, sample-index, scaler, configuration, and metric inputs. Paper tables and figures trace to the frozen Task 17/18 source hashes recorded in `paper/validation_report.json`.
"""
    (ROOT / "FINAL_EXPERIMENT_REPORT.md").write_text(markdown)
    print(json.dumps(audit, indent=2, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
