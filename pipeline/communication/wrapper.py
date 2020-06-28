import torch
from torch import Tensor
import torch.nn as nn

from functools import partial
from .util import flatten, unflatten, nested_map
from typing import Tuple, Union

filter_none = partial(filter, None)


class TensorWrapper:
    def __init__(self, structure):
        self.structure = structure
        self.flattened_structure = list(flatten(structure))
        self.flattenned_filtered_none = list(
            filter_none(self.flattened_structure))

    def tensors(self, x):
        ts = []

        for a in flatten(x):
            if a is None:
                ts.append(torch.tensor())
            elif isinstance(a, Tensor):
                ts.append(a)
            else:
                t = torch.tensor(a)
                ts.append(t)

        return ts

    def reconstruct_activations(self, x):

        ts = []
        ix = iter(x)
        for fs in self.flattened_structure:
            if fs is None:
                ts.append(None)
            else:
                ts.append(next(ix))
        return unflatten(ts, self.structure)

    def reconstruct_gradients(self, x):
        # None are not sent.
        assert (len(x) == len(self.flattenned_filtered_none))
        return unflatten(x, self.structure)
