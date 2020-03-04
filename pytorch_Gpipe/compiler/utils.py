import torch


def format_shape(shape):
    if isinstance(shape, torch.Size):
        return list(shape)
    return tuple(map(format_shape, shape))
