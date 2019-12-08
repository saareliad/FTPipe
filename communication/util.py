# Utils
import os
from enum import Enum, auto
from collections import deque


def get_world_size() -> int:
    """Returns world size (from env), or 1 if not set"""
    return int(os.environ.get('WORLD_SIZE', 1))


def get_global_rank() -> int:
    """Returns global rank (from env), or 0 if not set"""
    return int(os.environ.get('RANK', 0))


class CommPolicy(Enum):
    P2P = auto()
    BCAST = auto()


def toPolicy(backend, cpu):
    assert backend in {'nccl', 'gloo', 'mpi'}

    if backend == 'mpi' or cpu:
        return CommPolicy.P2P

    return CommPolicy.BCAST


def createBufferConfigs(xs, partitionConfig):
    '''
    performs a forward pass of the partitioned model and records the size and dtype of every data transfer

    Parameters:
    -----------
    xs:
        the input for the model

    partitionConfig:
        the configuration we generated, aka the output of createConfig()

    Return:
    -------
    dictionary from tensor name to {size,dtype}
    '''
    nparts = len(i for i in partitionConfig if isinstance(i, int))
    bufferConfigs = {}

    ts = dict(zip(partitionConfig['model inputs'], xs))

    for n, t in ts.items():
        bufferConfigs[n] = {'size': t.shape, 'dtype': t.dtype}

    parts = deque(range(nparts))
    # here we assume a DAG structure and not sequential structure
    while parts:
        idx = parts.popleft()
        partition = partitionConfig[idx]
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

        outs = model(*inputs)
        # update outputs
        for n, o in zip(partition['outputs'], outs):
            ts[n] = o
            bufferConfigs[n] = {'size': o.shape,
                                'dtype': o.dtype}

    for n, t in zip(partitionConfig['model outputs'], outs):
        bufferConfigs[n] = {'size': t.shape, 'dtype': t.dtype}

    return bufferConfigs
