# VA-Diff ETH

Reproducible research code for **Volatility-Aware Diffusion Models for Cryptocurrency Return Forecasting** using ETH/USD hourly returns.

## Research question
Does explicit realized-volatility conditioning improve probabilistic diffusion forecasting, especially in high-volatility regimes?

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/smoke_test.py
pytest -q
```

## Experiment contract
- Data: ETH/USD hourly OHLCV
- Context: 168 hours
- Horizons: 1, 6, 24 hours
- Volatility: 24-hour realized volatility
- Main comparison: Diffusion vs VA-Diff
- Primary metric: CRPS
- Regimes: low / medium / high volatility

## Volatility feature policy

`rv_24h` is the only primary volatility feature for main-manuscript experiments. It is
the default model-conditioning feature and the feature from which regime thresholds
are fit using training data only.

`rv_168h` remains available in the feature dataset as a diagnostic/sensitivity feature
only. It must not be used for primary model conditioning, main regime definitions,
train/validation/test regime thresholds, headline result tables, model selection, or
hyperparameter tuning. It may be used after the main experiment for descriptive
diagnostics, robustness checks, or optional supplementary figures/tables.

The feature-selection guard rejects `rv_168h` in the main/default mode. A caller must
explicitly enable `sensitivity_mode` for a diagnostic or sensitivity analysis.

See `AGENTS.md` for Codex instructions and `configs/base.yaml` for defaults.
