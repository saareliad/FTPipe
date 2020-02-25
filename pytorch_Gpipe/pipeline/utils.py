import numpy as np
from enum import Enum, auto, unique


class InvalidState(Exception):
    ''' Error used to indicate that the pipeline is not in the correct state
    for some operation for eg. backward when in eval mode will raise this exception
    '''


def split_to_n(l, n):
    '''
    return a list of n even chunks of l 
    '''
    sizes = np.full(n, len(l) // n)
    sizes[:len(l) % n] += 1
    ends = np.cumsum(sizes)

    return[l[ends[i] - sizes[i]:ends[i]] for i in range(len(sizes))]


@unique
class SyncBuffersMode(Enum):
    DISABLED = auto()
    # sync buffers only when switching to eval model during training
    BEFORE_EVAL = auto()
