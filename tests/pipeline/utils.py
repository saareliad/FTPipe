
import torch


def tensors_almost_equal(a, b):
    return torch.all(torch.lt(torch.abs(a - b), 1e-12)).item() == 1