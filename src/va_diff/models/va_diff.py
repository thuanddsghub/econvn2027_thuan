import torch
from torch import nn


class VADiffDenoiser(nn.Module):
    """Minimal conditional denoiser: noisy future + context + realized volatility + diffusion step."""

    def __init__(self, horizon: int, context_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.GRU(input_size=1, hidden_size=context_dim, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(horizon + context_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, noisy_future, history, volatility, diffusion_step):
        _, h = self.encoder(history.unsqueeze(-1))
        context = h[-1]
        x = torch.cat([noisy_future, context, volatility[:, None], diffusion_step[:, None]], dim=1)
        return self.net(x)
