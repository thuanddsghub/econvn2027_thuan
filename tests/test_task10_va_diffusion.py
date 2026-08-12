import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fit_task10_va_diffusion import tiny_overfit_check
from va_diff.models.standard_diffusion import StandardConditionalDiffusion
from va_diff.models.va_diffusion import (
    VAGaussianDiffusion,
    VolatilityAwareConditionalDiffusion,
    build_diffusion_model,
)


def test_disabled_conditioning_returns_standard_architecture() -> None:
    torch.manual_seed(42)
    standard = StandardConditionalDiffusion(horizon=6, hidden_size=32, gru_layers=1)
    torch.manual_seed(42)
    disabled = build_diffusion_model(6, use_volatility_conditioning=False)
    assert isinstance(disabled, StandardConditionalDiffusion)
    assert not isinstance(disabled, VolatilityAwareConditionalDiffusion)
    assert all(torch.equal(standard.state_dict()[key], disabled.state_dict()[key]) for key in standard.state_dict())


def test_volatility_branch_is_separate_and_shapes_are_valid() -> None:
    model = VolatilityAwareConditionalDiffusion(horizon=24)
    history = torch.randn((2, 168, 4))
    noisy = torch.randn((2, 24))
    volatility = torch.randn((2, 1))
    output = model(noisy, history, torch.tensor([0, 99]), volatility)
    assert output.shape == (2, 24)
    assert hasattr(model, "volatility_branch")
    assert "rv_168h" not in str(model)


def test_va_tiny_problem_can_be_overfit() -> None:
    result = tiny_overfit_check()
    assert result["status"]
    assert result["final_loss"] < result["initial_loss"]


def test_va_reverse_sampling_is_finite_and_reproducible() -> None:
    torch.manual_seed(42)
    model = VolatilityAwareConditionalDiffusion(horizon=6)
    diffusion = VAGaussianDiffusion()
    history = torch.randn((2, 168, 4))
    volatility = torch.randn((2, 1))
    first = diffusion.sample(model, history, volatility, sample_count=2, generator=torch.Generator().manual_seed(123))
    second = diffusion.sample(model, history, volatility, sample_count=2, generator=torch.Generator().manual_seed(123))
    assert first.shape == (2, 2, 6)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_saved_va_diffusion_artifacts_pass_protocol() -> None:
    report = json.loads(Path("outputs/metadata/task10_validation_report.json").read_text())
    metadata = json.loads(Path("outputs/metadata/task10_model_metadata.json").read_text())
    metrics = pd.read_csv("outputs/metrics/task10_va_diffusion_metrics.csv")
    assert report["status"] == "PASS"
    assert report["use_volatility_conditioning"]
    assert report["rv_168h_used"] is False
    assert metadata["volatility_source"] == "scaled rv_24h at endpoint timestamp t only"
    assert set(metrics["horizon"]) == {1, 6, 24}
    archive = np.load("outputs/predictions/task10_va_diffusion_samples/test_h24.npz")
    assert archive["samples"].shape[1:] == (100, 24)
    assert np.isfinite(archive["samples"]).all()
