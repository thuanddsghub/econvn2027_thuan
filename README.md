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

See `AGENTS.md` for Codex instructions and `configs/base.yaml` for defaults.
