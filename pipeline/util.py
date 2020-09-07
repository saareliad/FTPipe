# Utils
import os
from enum import Enum, auto
from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor


def get_world_size(backend) -> int:
    """Returns world size (from env), or 1 if not set"""
    if backend != 'mpi':
        return int(os.environ.get('WORLD_SIZE', 1))
    else:
        return int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))


def get_global_rank(backend) -> int:
    """Returns global rank (from env), or 0 if not set"""
    if backend != 'mpi':
        return int(os.environ.get('RANK', 0))
    else:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))


class CommPolicy(Enum):
    P2P = auto()
    BCAST = auto()


def to_policy(backend, cpu):
    assert backend in {'nccl', 'gloo', 'mpi'}

    if backend == 'mpi' or cpu:
        return CommPolicy.P2P

    return CommPolicy.BCAST


def nested_map(func, ts, full=False):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        return func(ts)
    elif isinstance(ts, (list, tuple, set)):
        return type(ts)(nested_map(func, t, full=full) for t in ts)
    elif isinstance(ts, dict):
        return {k: nested_map(func, v, full=full) for k, v in ts.items()}
    elif isinstance(ts, slice) and full:
        start = nested_map(func, ts.start, full=full)
        stop = nested_map(func, ts.stop, full=full)
        step = nested_map(func, ts.step, full=full)
        return slice(start, stop, step)
    return func(ts)


def flatten(x: Any) -> List[Any]:
    r"""Returns a flattened list of objects from a nested structure."""
    l: List[Any] = []
    if isinstance(x, torch.Size):
        l.append(x)
    elif isinstance(x, dict):
        # sorted(x.items(), key=lambda t: t[0])
        for y in x.values():
            l.extend(flatten(y))
    elif isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
        for y in x:
            l.extend(flatten(y))
    else:
        l.append(x)
    return l


def unflatten(xs, structure):
    res = _unflatten(xs, structure)[0]
    f_xs = list(flatten(xs))
    f_res = list(flatten(res))
    assert (len(f_xs) == len(f_res))
    return res


def _unflatten(xs, structure):
    if isinstance(structure, torch.Size):
        # torch.Size is subclass of tuple which is stupid
        return xs[0], 1

    if not isinstance(structure, (list, tuple, set, dict)):
        return xs[0], 1

    if isinstance(structure, (list, tuple, set)):
        offset = 0
        elements = []
        for s in structure:
            e, n = _unflatten(xs[offset:], s)
            elements.append(e)
            offset += n

        return type(structure)(elements), offset

    assert isinstance(structure, dict)
    offset = 0
    elements = dict()
    for k, v in sorted(structure.items(), key=lambda t: t[0]):
        e, n = _unflatten(xs[offset:], v)
        elements[k] = e
        offset += n

    return elements, offset


# def expand_like(val, structure):
#     return unflatten([val for _ in flatten(structure)], structure)


def detach_tensors(ts):
    def detach_if_tensor(t):
        if isinstance(t, Tensor):
            # NOTE: it is required for shared stateless!
            # to Set requires grad like the tensor.
            # especially if isinstance(x, torch.nn.Parameter)
            return t.detach().requires_grad_(t.requires_grad)
        return t

    return nested_map(detach_if_tensor, ts)


def move_tensors(ts, device):
    def move(t):
        if isinstance(t, (nn.Module, Tensor)):
            return t.to(device)
        return t

    return nested_map(move, ts)


# helper for debugging
def print_tensors(stage, x, in_or_out="out"):
    if isinstance(x, torch.Tensor):
        pass
    else:
        t = []
        for i, v in enumerate(x):
            if not isinstance(v, torch.Tensor):
                t.append("non-tensor" + str(v))
            else:
                t.append(v.shape)
        print(f"stage {stage}: {in_or_out}: {t}")
