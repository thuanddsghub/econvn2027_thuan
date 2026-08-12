"""Explicit-volatility extension of the standard conditional diffusion model."""

from __future__ import annotations

import torch
from torch import nn

from va_diff.models.standard_diffusion import GaussianDiffusion, StandardConditionalDiffusion


class VolatilityAwareConditionalDiffusion(StandardConditionalDiffusion):
    """Standard model plus a separate forecast-origin volatility embedding."""

    def __init__(self, horizon: int, *, volatility_dimension: int = 32) -> None:
        super().__init__(horizon=horizon, input_features=4, hidden_size=32, gru_layers=1)
        self.volatility_branch = nn.Sequential(
            nn.Linear(1, volatility_dimension),
            nn.SiLU(),
            nn.Linear(volatility_dimension, volatility_dimension),
        )
        self.denoiser = nn.Sequential(
            nn.Linear(horizon + 32 + 32 + volatility_dimension, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, horizon),
        )

    def denoise_from_context(
        self,
        noisy_future: torch.Tensor,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        timestep = self.timestep_embedding(timesteps)
        volatility_embedding = self.volatility_branch(volatility.reshape(-1, 1))
        return self.denoiser(
            torch.cat([noisy_future, context, timestep, volatility_embedding], dim=1)
        )

    def forward(
        self,
        noisy_future: torch.Tensor,
        history: torch.Tensor,
        timesteps: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoise_from_context(
            noisy_future, self.encode_context(history), timesteps, volatility
        )


class VAGaussianDiffusion(GaussianDiffusion):
    """DDPM operations with the additional origin-volatility argument."""

    def loss(
        self,
        model: VolatilityAwareConditionalDiffusion,
        clean: torch.Tensor,
        history: torch.Tensor,
        volatility: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = torch.randint(0, self.steps, (clean.shape[0],), device=clean.device)
        noisy, noise = self.q_sample(clean, timesteps)
        context = model.encode_context(history)
        prediction = model.denoise_from_context(noisy, context, timesteps, volatility)
        return torch.mean((prediction - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        model: VolatilityAwareConditionalDiffusion,
        history: torch.Tensor,
        volatility: torch.Tensor,
        *,
        sample_count: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        context = model.encode_context(history).repeat_interleave(sample_count, dim=0)
        expanded_volatility = volatility.repeat_interleave(sample_count, dim=0)
        current = torch.randn(
            context.shape[0], model.horizon, device=history.device, generator=generator
        )
        model.eval()
        for step in range(self.steps - 1, -1, -1):
            timesteps = torch.full(
                (current.shape[0],), step, device=history.device, dtype=torch.long
            )
            epsilon = model.denoise_from_context(
                current, context, timesteps, expanded_volatility
            )
            beta = self.betas[step].to(current.device)
            alpha = self.alphas[step].to(current.device)
            alpha_bar = self.alpha_bars[step].to(current.device)
            mean = (current - beta / (1.0 - alpha_bar).sqrt() * epsilon) / alpha.sqrt()
            if step > 0:
                noise = torch.randn(current.shape, device=current.device, generator=generator)
                current = mean + beta.sqrt() * noise
            else:
                current = mean
        return current.reshape(history.shape[0], sample_count, model.horizon)


def build_diffusion_model(horizon: int, use_volatility_conditioning: bool) -> nn.Module:
    """Return the standard architecture when conditioning is disabled."""
    if use_volatility_conditioning:
        return VolatilityAwareConditionalDiffusion(horizon=horizon)
    return StandardConditionalDiffusion(horizon=horizon, hidden_size=32, gru_layers=1)
