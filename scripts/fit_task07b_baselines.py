"""Fit fixed statistical baselines and evaluate them on Task 06 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

HORIZONS = (1, 6, 24)
MODELS = ("zero_return", "arima_100", "garch_110_ar1")
VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2025-01-01T00:00:00Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_forecast(last_return: float, horizon: int) -> np.ndarray:
    del last_return
    return np.zeros(horizon, dtype=float)


def recursive_ar_forecast(last_returns: np.ndarray, horizons: np.ndarray, intercept: float, phi: float) -> np.ndarray:
    powers = np.power(phi, horizons)
    if np.isclose(phi, 1.0):
        return last_returns * powers + intercept * horizons
    return last_returns * powers + intercept * (1.0 - powers) / (1.0 - phi)


def fit_arima(train_returns: np.ndarray) -> tuple[dict[str, float], Callable[[float, int], np.ndarray]]:
    model = ARIMA(train_returns, order=(1, 0, 0), trend="c")
    result = model.fit()
    params = dict(zip(result.param_names, result.params))
    intercept = float(params["const"])
    phi = float(params["ar.L1"])

    def forecast(last_return: float, horizon: int) -> np.ndarray:
        current = last_return
        values = []
        for _ in range(horizon):
            current = intercept + phi * current
            values.append(current)
        return np.asarray(values, dtype=float)

    return {"intercept": intercept, "ar1": phi}, forecast


def fit_garch(train_returns: np.ndarray) -> tuple[dict[str, float], Callable[[float, int], np.ndarray]]:
    # Fit in percent units for numerical conditioning; convert mean forecasts back to returns.
    scaled = train_returns * 100.0
    model = arch_model(
        scaled,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False,
    )
    result = model.fit(disp="off")
    params = dict(zip(result.params.index, result.params.to_numpy()))
    intercept = float(params["Const"]) / 100.0
    phi = float(params["y[1]"])

    def forecast(last_return: float, horizon: int) -> np.ndarray:
        current = last_return
        values = []
        for _ in range(horizon):
            current = intercept + phi * current
            values.append(current)
        return np.asarray(values, dtype=float)

    return {
        "mean_intercept_original_units": intercept,
        "mean_ar1": phi,
        "volatility_model": "GARCH(1,1) with normal innovations",
        "volatility_forecast_used_as_return_prediction": False,
    }, forecast


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for (model, split, horizon), group in predictions.groupby(
        ["model", "split", "horizon"], sort=False
    ):
        error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        grouped.append(
            {
                "model": model,
                "split": split,
                "horizon": horizon,
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    return pd.DataFrame(grouped, columns=["model", "split", "horizon", "n", "mae", "rmse"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/eth_usd_hourly_features.parquet"))
    parser.add_argument("--sample-index-path", type=Path, default=Path("data/processed/task06_sample_index.parquet"))
    parser.add_argument("--task06-manifest-path", type=Path, default=Path("data/metadata/task06_dataset_manifest.json"))
    parser.add_argument("--scaler-path", type=Path, default=Path("data/metadata/task07a_scaler.json"))
    parser.add_argument("--regime-index-path", type=Path, default=Path("data/processed/task05_split_index.parquet"))
    parser.add_argument("--prediction-path", type=Path, default=Path("outputs/predictions/task07b_predictions.parquet"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/task07b_metrics.csv"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata/task07b_model_metadata.json"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/metadata/task07b_validation_report.json"))
    args = parser.parse_args()

    features_sha256 = sha256(args.features_path)
    sample_index_sha256 = sha256(args.sample_index_path)
    scaler_sha256 = sha256(args.scaler_path)
    features = pd.read_parquet(args.features_path)
    sample_index = pd.read_parquet(args.sample_index_path)
    regime_index = pd.read_parquet(args.regime_index_path)
    scaler_artifact = json.loads(args.scaler_path.read_text(encoding="utf-8"))
    task06_manifest = json.loads(args.task06_manifest_path.read_text(encoding="utf-8"))
    if task06_manifest["sample_index_sha256"] != sample_index_sha256:
        raise ValueError(
            "Task 06 sample index checksum does not match its manifest; refusing stale cached definitions"
        )
    if scaler_artifact["scaler"]["targets_scaled"] is not False:
        raise ValueError("Task 07A scaler artifact does not guarantee unscaled targets")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    sample_index["endpoint_timestamp"] = pd.to_datetime(sample_index["endpoint_timestamp"], utc=True)
    regime_index["timestamp"] = pd.to_datetime(regime_index["timestamp"], utc=True)
    regime_by_timestamp = regime_index.set_index("timestamp")["regime"]
    if not sample_index["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("unknown split in Task 06 sample index")
    if sample_index[["split", "horizon", "endpoint_row"]].duplicated().any():
        raise ValueError("duplicate Task 06 sample definitions")
    if "rv_168h" not in features.columns:
        raise ValueError("expected feature dataset schema with diagnostic rv_168h column")

    train_returns = features.loc[features["timestamp"] < VALIDATION_START, "log_return"].dropna().to_numpy(dtype=float)
    if not np.isfinite(train_returns).all():
        raise ValueError("non-finite train returns")
    arima_metadata, _arima_forecast = fit_arima(train_returns)
    garch_metadata, _garch_forecast = fit_garch(train_returns)
    endpoints = sample_index["endpoint_row"].to_numpy(dtype=np.int64)
    horizons = sample_index["horizon"].to_numpy(dtype=np.int64)
    target_positions = endpoints + horizons
    last_returns = features["log_return"].to_numpy(dtype=float)[endpoints]
    targets = features["log_return"].to_numpy(dtype=float)[target_positions]
    if not np.isfinite(targets).all() or not np.isfinite(last_returns).all():
        raise ValueError("Task 06 sample contains a non-finite target or forecast-origin return")
    target_timestamps = features["timestamp"].to_numpy()[target_positions]
    regimes = regime_by_timestamp.reindex(sample_index["endpoint_timestamp"]).to_numpy()
    base = pd.DataFrame(
        {
            "timestamp": target_timestamps,
            "origin_timestamp": sample_index["endpoint_timestamp"].to_numpy(),
            "split": sample_index["split"].to_numpy(),
            "horizon": horizons,
            "y_true": targets,
            "regime": regimes,
            "config_model_metadata_ref": "outputs/metadata/task07b_model_metadata.json",
        }
    )
    prediction_frames = []
    for model_name in MODELS:
        if model_name == "zero_return":
            predictions_for_model = np.zeros(len(base), dtype=float)
        elif model_name == "arima_100":
            predictions_for_model = recursive_ar_forecast(
                last_returns, horizons, arima_metadata["intercept"], arima_metadata["ar1"]
            )
        else:
            predictions_for_model = recursive_ar_forecast(
                last_returns,
                horizons,
                garch_metadata["mean_intercept_original_units"],
                garch_metadata["mean_ar1"],
            )
        if not np.isfinite(predictions_for_model).all():
            raise ValueError(f"non-finite prediction from {model_name}")
        model_frame = base.copy()
        model_frame["model"] = model_name
        model_frame["y_pred"] = predictions_for_model
        prediction_frames.append(model_frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions["origin_timestamp"] = pd.to_datetime(predictions["origin_timestamp"], utc=True)
    predictions = predictions[
        [
            "timestamp",
            "origin_timestamp",
            "split",
            "horizon",
            "model",
            "y_true",
            "y_pred",
            "regime",
            "config_model_metadata_ref",
        ]
    ]
    metrics = metric_rows(predictions)
    sample_counts = {
        f"{row.model}/{row.split}/horizon_{row.horizon}": int(row.n)
        for row in metrics.itertuples(index=False)
    }
    args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.prediction_path, index=False, engine="pyarrow")
    metrics.to_csv(args.metrics_path, index=False)
    model_metadata = {
        "task": "07B",
        "models": {
            "zero_return": {"specification": "y_pred(t,H) = 0"},
            "arima_100": {
                "specification": "ARIMA(1,0,0) with constant; fitted on train log_return only",
                "parameters": arima_metadata,
            },
            "garch_110_ar1": {
                "specification": "AR(1)-GARCH(1,1), normal innovations, fitted on train log_return only",
                "parameters": garch_metadata,
                "mean_forecast_is_y_pred": True,
                "volatility_forecast_is_not_y_pred": True,
            },
        },
        "selection_policy": "fixed specifications; no validation/test tuning",
        "target_scale": "original unscaled log-return units",
        "scaler_artifact_reference": str(args.scaler_path),
        "scaler_sha256": scaler_sha256,
        "features_sha256": features_sha256,
        "sample_index_sha256": sample_index_sha256,
        "task06_manifest_sample_index_sha256": task06_manifest["sample_index_sha256"],
        "prediction_schema": list(predictions.columns),
    }
    args.metadata_path.write_text(json.dumps(model_metadata, indent=2) + "\n", encoding="utf-8")
    expected_per_definition = len(sample_index)
    validation = {
        "status": "PASS",
        "features_sha256": features_sha256,
        "sample_index_sha256": sample_index_sha256,
        "task06_manifest_sample_index_sha256": task06_manifest["sample_index_sha256"],
        "authoritative_task06_sample_index_verified": True,
        "scaler_sha256": scaler_sha256,
        "zero_return_exact": bool((predictions.loc[predictions.model == "zero_return", "y_pred"] == 0).all()),
        "identical_target_definitions": bool(predictions.groupby(["split", "horizon", "timestamp"])["model"].nunique().eq(3).all()),
        "no_nan_inf_evaluated_rows": bool(np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all()),
        "prediction_rows": len(predictions),
        "expected_prediction_rows": expected_per_definition * len(MODELS),
        "sample_counts": sample_counts,
        "test_used_for_tuning": False,
        "rv_168h_used": False,
        "targets_unscaled": True,
        "metrics_recomputed_from_saved_predictions": True,
    }
    validation["status"] = "PASS" if all(
        [
            validation["zero_return_exact"],
            validation["identical_target_definitions"],
            validation["no_nan_inf_evaluated_rows"],
            validation["prediction_rows"] == validation["expected_prediction_rows"],
            validation["authoritative_task06_sample_index_verified"],
            validation["test_used_for_tuning"] is False,
            validation["rv_168h_used"] is False,
            validation["targets_unscaled"],
        ]
    ) else "FAIL"
    args.report_path.write_text(json.dumps({"schema_version": "task07b.v1", "validation": validation}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": model_metadata, "metrics": metrics.to_dict(orient="records"), "validation": validation}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
