# Utils
import os
from enum import Enum, auto
from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
from itertools import chain


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


def create_buffer_configs(xs, partitions_config):
    '''
    performs a forward pass of the partitioned model and records the size and dtype of every data transfer

    Parameters:
    -----------
    xs:
        the input for the model

    partitions_config:
        the configuration we generated, aka the output of createConfig()

    Return:
    -------
    dictionary from tensor name to {size,dtype}
    '''
    if not isinstance(xs, tuple):
        xs = (xs, )
    nparts = len([i for i in partitions_config if isinstance(i, int)])
    buffer_configs = dict()
    ts = dict(zip(partitions_config['model inputs'], xs))

    for n, t in ts.items():
        buffer_configs[n] = {'size': t.shape, 'dtype': t.dtype}

    parts = deque(range(nparts))
    # here we assume a DAG structure and not sequential structure
    while parts:
        idx = parts.popleft()
        partition = partitions_config[idx]
        model = partition['model']
        # gather inputs
        inputs = []
        for n in partition['inputs']:
            if not (n in ts):
                break
            else:
                inputs.append(ts[n])

        # not all inputs were ready proceed to next partition and try again later
        if len(inputs) < len(partition['inputs']):
            parts.append(idx)
            continue

        # move inputs to the model device and forward pass
        device = list(model.parameters())[0].device
        inputs = [t.to(device) for t in inputs]
        outs = model(*inputs)

        # update outputs
        for n, o in zip(partition['outputs'], outs):
            ts[n] = o
            buffer_configs[n] = {'size': o.shape, 'dtype': o.dtype}

    for n, t in zip(partitions_config['model outputs'], outs):
        buffer_configs[n] = {'size': t.shape, 'dtype': t.dtype}

    return buffer_configs


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


def flatten(ts):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        yield ts
    elif isinstance(ts, (list, tuple, set)):
        yield from chain(*[flatten(t) for t in ts])
    elif isinstance(ts, dict):
        yield from chain(
            *[flatten(t) for k, t in sorted(ts.items(), key=lambda t: t[0])])
    else:
        yield ts


def unflatten(xs, structure):
    return _unflatten(xs, structure)[0]


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
