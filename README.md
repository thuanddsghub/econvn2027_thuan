# Volatility-Aware Diffusion Models for Cryptocurrency Return Forecasting

This repository is frozen for the manuscript experiment. No additional model
training, hyperparameter tuning, recalibration, or sensitivity analysis is part of
the active workflow.

## Manuscript scope

The final comparison is Standard Diffusion versus VA-Diff for hourly ETH/USD return
forecasting at H=1, 6, and 24. The authoritative manuscript artifacts are under
`outputs/final/` and are audited by Task 20.

- 168-hour chronological context
- train/validation/test chronological splits
- train-only input scaling and train-only target standardization
- corrected DDPM posterior variance
- 100 trajectories per forecast
- five paired seeds: 42, 123, 2024, 3407, 7777
- primary metric: CRPS; secondary metrics: MAE, RMSE, PICP90, interval width
- frozen train-only `rv_24h` regimes

`rv_168h` is retained only as a diagnostic field in the feature dataset and is not
used by the manuscript pipeline.

## Reproduce the audit

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make paper-audit
pytest -q
ruff check .
```

The audit does not retrain models or regenerate tables and figures. It verifies the
frozen hashes, sample counts, scaling policy, regime thresholds, paper artifacts,
paired statistics, and manuscript-safe conclusions.

## Manuscript draft

The manuscript structure and current evidence are in
[`manuscript/manuscript.md`](manuscript/manuscript.md). Tables and figures are in
[`outputs/final/paper/`](outputs/final/paper/).
