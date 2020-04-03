import torch


def format_shape_or_dtype(shape_or_dtype):
    if isinstance(shape_or_dtype,torch.Size):
        return list(shape_or_dtype)
    elif isinstance(shape_or_dtype,torch.dtype):
        return shape_or_dtype
    return tuple(map(format_shape_or_dtype, shape_or_dtype))
