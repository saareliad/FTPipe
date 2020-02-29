import numpy as np
import torch
from torch import Tensor
from enum import Enum, auto, unique
from typing import Iterable, Tuple


class InvalidState(Exception):
    ''' Error used to indicate that the pipeline is not in the correct state
    for some operation for eg. backward when in eval mode will raise this exception
    '''


def list_chunk(l: Iterable, n: int) -> Tuple[Iterable, ...]:
    '''
    return a list of n even chunks of l 
    '''
    sizes = np.full(n, len(l) // n)
    sizes[:len(l) % n] += 1
    ends = np.cumsum(sizes)

    return tuple(l[ends[i] - sizes[i]:ends[i]] for i in range(len(sizes)))


def tensor_chunk(t: Tensor, n: int, dim: int = 0) -> Tuple[Tensor, ...]:
    if t is None:
        return [None] * n
    sizes = torch.full((n,), t.size(dim) // n, dtype=torch.int32)
    sizes[:t.size(dim) % n] += 1
    return torch.split(t, sizes.tolist(), dim=dim)


@unique
class SyncBuffersMode(Enum):
    DISABLED = auto()
    # sync buffers only when switching to eval model during training
    BEFORE_EVAL = auto()
