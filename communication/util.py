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


def createBufferConfigs(xs, partitions_config):
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
        xs = (xs,)
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
            buffer_configs[n] = {'size': o.shape,
                                 'dtype': o.dtype}

    for n, t in zip(partitions_config['model outputs'], outs):
        buffer_configs[n] = {'size': t.shape, 'dtype': t.dtype}

    return buffer_configs
