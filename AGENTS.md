# Codex instructions

Project goal: reproducible experiments for Volatility-Aware Diffusion Models for ETH/USD return forecasting.

Rules:
- Never use random train/test splits; preserve chronological order.
- Fit scalers and regime thresholds on training data only.
- Primary metric is CRPS; also report MAE, RMSE, and 90% interval coverage.
- Compare standard conditional Diffusion vs VA-Diff with identical capacity/training settings.
- Every experiment must be config-driven and seeded.
- Save config, metrics, checkpoints, and figures under outputs/<experiment_name>/.
- Add or update tests when modifying data transforms, metrics, or model shapes.
- Do not silently download datasets during training; data acquisition is a separate script.
