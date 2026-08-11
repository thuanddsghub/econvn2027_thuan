import random

import numpy as np
import torch

from va_diff.utils.repro import seed_everything


def _random_outputs(seed: int) -> tuple[float, float, torch.Tensor]:
    seed_everything(seed)
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_seed_reproduces_random_outputs() -> None:
    first = _random_outputs(1234)
    second = _random_outputs(1234)

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
