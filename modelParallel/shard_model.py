import numpy as np
import torch.nn as nn


# split the given list to num_splits even sub lists
# optional perform func on each sub list
def split_to_n(to_splilt, num_splits, func=None):
    sizes = np.full(num_splits, len(to_splilt) // num_splits)
    sizes[:len(to_splilt) % num_splits] += 1
    ends = np.cumsum(sizes)
    if func is None:
        return[to_splilt[ends[i]-sizes[i]:ends[i]] for i in range(len(sizes))]
    else:
        return[func(*to_splilt[ends[i]-sizes[i]:ends[i]]) for i in range(len(sizes))]


def build_shards(model, num_shards, func=nn.Sequential):
    return nn.ModuleList(split_to_n(
        list(model.children()), num_shards, func))
