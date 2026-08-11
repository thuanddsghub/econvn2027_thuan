import torch
from va_diff.models.va_diff import VADiffDenoiser


def test_output_shape():
    model = VADiffDenoiser(horizon=6)
    y = model(torch.randn(2, 6), torch.randn(2, 168), torch.rand(2), torch.rand(2))
    assert y.shape == (2, 6)
