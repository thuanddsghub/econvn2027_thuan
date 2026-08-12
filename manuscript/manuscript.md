# Volatility-Aware Diffusion Models for Cryptocurrency Return Forecasting

## Abstract

Diffusion models offer a flexible approach to probabilistic forecasting, but their behavior on financial returns can be dominated by implementation and scaling choices. We evaluate a standard conditional diffusion model and a volatility-aware variant (VA-Diff) for hourly ETH/USD return forecasting at 1-, 6-, and 24-hour horizons. The experimental pipeline uses chronological splits, train-only feature and target scaling, corrected DDPM posterior variance, 100 Monte Carlo trajectories per forecast, and five paired random seeds. Target standardization materially reduces predictive over-dispersion and improves calibration. In contrast, explicit conditioning on 24-hour realized volatility does not provide a robust CRPS advantage: mean improvements are only +0.088% at H=1 and +0.062% at H=6, while performance deteriorates by 0.364% at H=24. Regime-level results also show no consistent gain in high-volatility periods. The study therefore highlights implementation correctness, calibration, and paired robustness analysis as prerequisites for credible diffusion-based financial forecasting.

**Keywords:** diffusion models; probabilistic forecasting; cryptocurrency; ETH/USD; realized volatility; calibration; CRPS

## 1. Introduction

Cryptocurrency returns are noisy, heavy-tailed, and subject to time-varying volatility. These properties make probabilistic forecasting more informative than single-point prediction, but they also make evaluation sensitive to the scale of the target, the handling of missing observations, and the stochastic sampling procedure. Diffusion models are attractive because they represent a full predictive distribution, yet the apparent quality of that distribution can be dominated by implementation details rather than by the conditioning mechanism under study.

This work studies whether explicit realized-volatility conditioning improves a conditional diffusion model for hourly ETH/USD returns. The comparison is intentionally controlled: Standard Diffusion and VA-Diff use the same data, windows, architecture capacity, optimizer, diffusion schedule, selection protocol, and sampling budget. VA-Diff adds only a separate conditioning branch for `rv_24h` at the forecast origin.

The contributions are:

1. A leakage-safe hourly ETH/USD pipeline that preserves the 38 missing candles and 18 contiguous gap groups.
2. An explicit audit of target parameterization and corrected DDPM reverse variance.
3. A five-seed paired comparison of Standard Diffusion and VA-Diff using CRPS as the primary metric.
4. A conservative interpretation of volatility-regime and calibration results that does not treat forecast timestamps as independent statistical replicates.

## 2. Methodology

### 2.1 Data and preprocessing

The dataset is ETH/USD hourly OHLCV data covering 2017-01-01 00:00 UTC through 2026-08-11 00:00 UTC. Timestamps are parsed as UTC and sorted chronologically. The raw series contains 38 missing hourly timestamps in 18 contiguous gaps. OHLCV values are not forward-filled. The log return is computed only when the current and previous observations are genuinely consecutive hourly candles; the first return and the 18 returns immediately following gap boundaries are therefore NaN.

The model features are `log_return`, `log_volume = log(1 + volume)`, `high_low_range = (high-low)/close`, and trailing realized volatility `rv_24h`:

$$
rv_{24h,t} = \sqrt{\sum_{i=0}^{23} r_{t-i}^2}.
$$

Windows containing an invalid return remain invalid. `rv_168h` is retained in the feature file for diagnostics only and is excluded from the primary experiment.

### 2.2 Chronological samples and regimes

The train split contains timestamps before 2024-01-01 UTC, validation covers 2024, and test begins at 2025-01-01 UTC. Contexts contain 168 consecutive hourly observations. Forecast targets are strictly after the context origin and remain inside the requested split. The authoritative Task 06 sample counts are:

| Split | H=1 | H=6 | H=24 |
|---|---:|---:|---:|
| Train | 58,125 | 58,045 | 57,757 |
| Validation | 8,784 | 8,779 | 8,761 |
| Test | 13,695 | 13,680 | 13,626 |

Volatility regimes use thresholds computed once from valid train `rv_24h` values: q33 = 0.0275758304 and q67 = 0.0454990582. The definitions are Low for values below q33, Medium for q33 through below q67, and High for values at or above q67. The same frozen thresholds are applied to validation and test.

### 2.3 Scaling and target parameterization

Input features are standardized with a feature-wise z-score fitted on train observations only. The diffusion target is also standardized using train log returns only:

$$
z_t = \frac{r_t-\mu_{train}}{\sigma_{train}}.
$$

The train target mean is 9.0353410e-05 and the train standard deviation is 0.0108063390. Diffusion training and reverse sampling operate in z-space. Samples are inverse-transformed to original log-return units before CRPS, MAE, RMSE, coverage, or interval-width evaluation. No future split contributes to either scaler.

### 2.4 Conditional diffusion models

Both models use a one-layer GRU with hidden size 32 to encode the 168-hour historical context. A timestep embedding and an MLP denoiser predict the diffusion noise for the future return vector at each horizon. Standard Diffusion uses the historical context and noisy future as inputs. VA-Diff adds a separate MLP branch receiving only the forecast-origin `rv_24h`; it does not receive future volatility.

The forward process uses 100 linearly scheduled diffusion steps with beta values from 1e-4 to 0.02. Reverse sampling uses the DDPM posterior variance:

$$
\widetilde{\beta}_t = \beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t},
$$

with no stochastic noise at t=0. Both models use batch size 512, learning rate 1e-3, weight decay 1e-5, a maximum of 10 epochs, validation-only checkpoint selection, and 100 trajectories per forecast.

## 3. Experimental setup

The experiment uses five paired seeds: 42, 123, 2024, 3407, and 7777. For every seed and horizon, Standard Diffusion and VA-Diff use identical sample timestamps, splits, optimizer settings, diffusion schedule, training budget, and sampling protocol. The only model difference is the explicit forecast-origin `rv_24h` conditioning branch.

CRPS is the primary metric. Secondary metrics are predictive-median MAE, predictive-median RMSE, 90% prediction interval coverage probability (PICP90), and interval width. Statistical tests are paired across seeds rather than pooled across forecast timestamps. Because n=5, p-values are reported as low-power exploratory diagnostics and are not used alone to claim significance.

## 4. Results and discussion

### 4.1 Overall performance

Mean test CRPS for Standard Diffusion and VA-Diff is 0.003540 and 0.003537 at H=1, 0.003586 and 0.003584 at H=6, and 0.003618 and 0.003632 at H=24, respectively. The corresponding paired mean changes in VA-Diff CRPS are -3.13e-06, -2.27e-06, and +1.32e-05. Thus, the short-horizon improvements are very small, while the 24-hour result favors Standard Diffusion.

The 90% interval coverage is approximately 0.856/0.861/0.887 for Standard Diffusion and 0.859/0.862/0.884 for VA-Diff at H=1/6/24. The target-standardized final pipeline avoids the severe over-dispersion identified during implementation diagnostics, but calibration is not perfect and is reported only at the frozen 90% level.

### 4.2 Volatility-regime results

Regime-level CRPS does not establish a consistent advantage for explicit volatility conditioning. At H=24, VA-Diff is worse in Low, Medium, and High regimes. At H=1 and H=6, the direction varies by regime and seed. In particular, high-volatility CRPS changes are positive for VA-Diff at all three horizons in the five-seed aggregate, so the manuscript does not claim a high-volatility gain.

### 4.3 Statistical robustness

For overall CRPS, paired t-test and Wilcoxon p-values are 0.3891/0.6250 at H=1, 0.7848/1.0000 at H=6, and 0.3123/0.3125 at H=24. These results do not support a robust VA-Diff advantage. The H=24 high-volatility comparison is unfavorable to VA-Diff, but the paired sample size remains too small for a strong statistical claim.

## 5. Conclusion and limitations

This study finds that implementation correctness and target parameterization materially affect diffusion forecasting behavior on cryptocurrency returns. Train-only target standardization and the corrected DDPM posterior variance produce a substantially more credible predictive scale and calibration profile. However, explicit conditioning on 24-hour realized volatility does not provide a reproducible CRPS advantage over a matched Standard Diffusion model.

The primary limitations are the single ETH/USD market, hourly resolution, five-seed robustness sample, and the frozen 90% calibration level. The results should therefore be interpreted as a controlled methodological comparison, not as evidence that volatility conditioning is universally ineffective. No additional model training, tuning, recalibration, or `rv_168h` sensitivity analysis is part of this manuscript.

## Reproducibility

The frozen manifest, metric tables, statistical summaries, paper tables, figures, and final audit are under `outputs/final/`. Their SHA-256 provenance is recorded in `final_manifest.json`, `paper/validation_report.json`, and `final_audit_report.json`.
