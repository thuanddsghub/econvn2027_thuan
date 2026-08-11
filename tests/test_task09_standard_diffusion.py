import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fit_task09_standard_diffusion import tiny_overfit_check
from va_diff.models.standard_diffusion import GaussianDiffusion, StandardConditionalDiffusion


def test_q_sample_matches_reference_formula() -> None:
    diffusion = GaussianDiffusion(steps=100, beta_start=1e-4, beta_end=0.02)
    clean = torch.tensor([[0.2, -0.1]])
    noise = torch.tensor([[0.5, -0.25]])
    timesteps = torch.tensor([37])
    sampled, returned_noise = diffusion.q_sample(clean, timesteps, noise=noise)
    alpha_bar = diffusion.alpha_bars[37]
    expected = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
    assert torch.allclose(sampled, expected)
    assert torch.equal(returned_noise, noise)


def test_diffusion_timestep_boundaries_and_horizon_shapes() -> None:
    diffusion = GaussianDiffusion()
    clean = torch.zeros((2, 24))
    for timestep in (0, diffusion.steps - 1):
        sampled, noise = diffusion.q_sample(clean, torch.full((2,), timestep), noise=torch.ones_like(clean))
        assert sampled.shape == clean.shape
        assert noise.shape == clean.shape
    for horizon in (1, 6, 24):
        model = StandardConditionalDiffusion(horizon=horizon)
        history = torch.randn((2, 168, 4))
        noisy = torch.randn((2, horizon))
        timestep = torch.tensor([0, 99])
        assert model(noisy, history, timestep).shape == (2, horizon)


def test_tiny_diffusion_problem_can_be_overfit() -> None:
    result = tiny_overfit_check()
    assert result["status"]
    assert result["final_loss"] < result["initial_loss"]


def test_reverse_sampling_is_finite_and_seed_reproducible() -> None:
    torch.manual_seed(42)
    model = StandardConditionalDiffusion(horizon=6)
    diffusion = GaussianDiffusion()
    history = torch.randn((2, 168, 4))
    first = diffusion.sample(model, history, sample_count=2, generator=torch.Generator().manual_seed(123))
    second = diffusion.sample(model, history, sample_count=2, generator=torch.Generator().manual_seed(123))
    assert first.shape == (2, 2, 6)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_saved_task09_artifacts_match_authoritative_protocol() -> None:
    sample_index = pd.read_parquet("data/processed/task06_sample_index.parquet")
    predictions = pd.read_parquet("outputs/predictions/task09_standard_diffusion_predictions.parquet")
    metadata = json.loads(Path("outputs/metadata/task09_model_metadata.json").read_text())
    report = json.loads(Path("outputs/metadata/task09_validation_report.json").read_text())
    assert metadata["features"] == ["log_return", "log_volume", "high_low_range", "rv_24h"]
    assert metadata["target"] == "original unscaled log_return"
    assert metadata["explicit_volatility_conditioning"] is False
    assert report["status"] == "PASS"
    assert report["all_168_context_observations_used"]
    assert len(predictions) == report["expected_prediction_rows"]
    assert np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all()
    for split in ("validation", "test"):
        for horizon in (1, 6, 24):
            expected = sample_index.loc[
                (sample_index["split"] == split) & (sample_index["horizon"] == horizon),
                "target_end_timestamp",
            ].astype("datetime64[ns, UTC]").sort_values().reset_index(drop=True)
            actual = predictions.loc[
                (predictions["split"] == split)
                & (predictions["horizon"] == horizon)
                & (predictions["target_step"] == horizon),
                "timestamp",
            ].drop_duplicates().astype("datetime64[ns, UTC]").sort_values().reset_index(drop=True)
            pd.testing.assert_series_equal(actual, expected, check_names=False)
