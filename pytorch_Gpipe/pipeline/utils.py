
from typing import Callable, Any
import torch

from pytorch_Gpipe.utils import Tensors, TensorsShape


def tensors_map(tensors: Tensors, f: Callable[[Tensors], Any]):
    """maps each tensor using the f function while keeping the same structure"""

    if isinstance(tensors, torch.Tensor):
        return f(tensors)

    container = type(tensors)
    return container([tensors_map(tensors, f) for tensors in tensors])


def tensors_bi_map(t1: Tensors, t2: Tensors, f: Callable[[Tensors, Tensors], Any]):
    """maps each pair of tensors using the f function"""

    if isinstance(t1, torch.Tensor):
        return f(t1, t2)

    container = type(t1)
    return container(tensors_bi_map(t1, t2, f) for t1, t2 in zip(t1, t2))


def get_devices(tensors: Tensors):
    """list of all the inner tensors device in order from left to right"""
    devices = []
    tensors_map(tensors, lambda tensor: devices.append(tensor.device))

    return devices


def tensors_to(tensors: Tensors, devices):
    """move each tensor to the the device in devices"""

    def move_tensor(tensor):
        for device in devices:
            yield tensor.to(device, non_blocking=True)

    tensors_map(tensors, move_tensor)


def gen_garbage_output(shape: TensorsShape, batch_dim, device) -> Tensors:
    """
    Generates empty Tensors.

    :param shape: the structure of the output Tensors
    :param batch_dim: The extra batch dimension to add the the contained tensors
    :param device: onto which to place the output
    :return: empty Tensors that match requested structure
    """
    # got to the inner-most shape, output is a tensor
    if isinstance(shape[0], int):
        return torch.empty(batch_dim, *shape, device=device)

    # choosing wrapper (tuple/list)
    container = type(shape)
    return container([gen_garbage_output(inner_shape, batch_dim, device) for inner_shape in shape])


def batch_dim(tensors: Tensors):
    """returns the batch_dim of a Tensors object"""
    if isinstance(tensors, torch.Tensor):
        return tensors.size(0)

    return batch_dim(tensors[0])
