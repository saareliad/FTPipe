
import torch


def tensors_almost_equal(a, b, epsilon=0.01):
    return torch.all(torch.lt(torch.abs(a - b), epsilon)).item() == 1
