import hydra
import torch

from va_diff.models.va_diff import VADiffDenoiser
from va_diff.utils.repro import seed_everything

seed_everything(42)
model = VADiffDenoiser(horizon=24)
b = 4
out = model(
    torch.randn(b, 24),
    torch.randn(b, 168),
    torch.rand(b),
    torch.rand(b),
)
assert out.shape == (b, 24)
print("smoke test: OK", tuple(out.shape))
print("hydra import: OK", hydra.__version__)
