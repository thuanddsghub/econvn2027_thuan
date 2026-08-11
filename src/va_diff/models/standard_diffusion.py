"""Standard conditional Gaussian diffusion for return-vector forecasting."""

from __future__ import annotations

import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return self.projection(embedding)


class StandardConditionalDiffusion(nn.Module):
    """GRU history encoder plus epsilon-prediction MLP denoiser."""

    def __init__(
        self,
        horizon: int,
        *,
        input_features: int = 4,
        hidden_size: int = 32,
        gru_layers: int = 1,
        timestep_dimension: int = 32,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.context_encoder = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.timestep_embedding = TimestepEmbedding(timestep_dimension)
        self.denoiser = nn.Sequential(
            nn.Linear(horizon + hidden_size + timestep_dimension, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, horizon),
        )

    def encode_context(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.context_encoder(history)
        return hidden[-1]

    def forward(
        self,
        noisy_future: torch.Tensor,
        history: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_context(history)
        return self.denoise_from_context(noisy_future, context, timesteps)

    def denoise_from_context(
        self,
        noisy_future: torch.Tensor,
        context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timestep = self.timestep_embedding(timesteps)
        return self.denoiser(torch.cat([noisy_future, context, timestep], dim=1))


class GaussianDiffusion:
    """Linear-beta DDPM schedule and epsilon-prediction operations."""

    def __init__(
        self,
        steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[timesteps].to(clean.device)
        while alpha_bar.ndim < clean.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def loss(
        self,
        model: StandardConditionalDiffusion,
        clean: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = torch.randint(0, self.steps, (clean.shape[0],), device=clean.device)
        noisy, noise = self.q_sample(clean, timesteps)
        return torch.mean((model(noisy, history, timesteps) - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        model: StandardConditionalDiffusion,
        history: torch.Tensor,
        *,
        sample_count: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        context = model.encode_context(history)
        context = context.repeat_interleave(sample_count, dim=0)
        current = torch.randn(
            context.shape[0], model.horizon, device=history.device, generator=generator
        )
        model.eval()
        for step in range(self.steps - 1, -1, -1):
            timesteps = torch.full(
                (current.shape[0],), step, device=history.device, dtype=torch.long
            )
            epsilon = model.denoise_from_context(current, context, timesteps)
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
