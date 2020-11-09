import os
import pickle

import torch

from .model_profiling.control_flow_graph import Graph


class TorchCache:
    def __init__(self, cache_name, overwrite=False):
        self.cache_name = cache_name
        self.exists = os.path.exists(cache_name)
        self.overwrite = overwrite
        self.v = None

    def __enter__(self):
        if self.exists:
            print(f"loading from cache: {self.cache_name}")
            self.v = torch.load(self.cache_name)
        else:
            print(f"computing value for {self.cache_name}")
        return self

    def __exit__(self, type, value, traceback):
        if not self.exists or self.overwrite:
            print(f"saving to cache: {self.cache_name}")
            assert self.v is not None, "You should enter a value"
            torch.save(self.v, self.cache_name)


class PickleCache:
    def __init__(self, cache_name, overwrite=False):
        self.cache_name = cache_name
        self.exists = os.path.exists(cache_name)
        self.overwrite = overwrite
        self.v = None

    def __enter__(self):
        if self.exists:
            print(f"loading from cache: {self.cache_name}")
            with open(self.cache_name, "rb") as f:
                self.v = pickle.load(f)
        else:
            print(f"computing value for {self.cache_name}")
        return self

    def __exit__(self, type, value, traceback):
        if not self.exists or self.overwrite:
            print(f"saving to cache: {self.cache_name}")
            assert self.v is not None, "You should enter a value"
            with open(self.cache_name, "wb") as f:
                pickle.dump(self.v, f)


class GraphCache:
    def __init__(self, cache_name, overwrite=False):
        self.cache_name = cache_name
        self.exists = os.path.exists(cache_name)
        self.overwrite = overwrite
        self.v = None

    def __enter__(self):
        if self.exists:
            print(f"loading from cache: {self.cache_name}")
            self.v = Graph.deserialize(self.cache_name)
        else:
            print(f"computing value for {self.cache_name}")
        return self

    def __exit__(self, type, value, traceback):
        if not self.exists or self.overwrite:
            print(f"saving to cache: {self.cache_name}")
            assert self.v is not None, "You should enter a value"
            assert isinstance(self.v, Graph)
            self.v.serialize(self.cache_name)


def compute_and_cache(compute_function, cache_name, *args, _cache_cls_to_use=TorchCache, **kw):
    """
    Compute or load from cache, optionally save results to cache.
    Return computed value
    Examples:
        # compute big
        # compute_and_cache(lambda: torch.ones(10), "big")
        # compute big, then small
        # compute_and_cache(lambda: torch.randn(10) * compute_and_cache(lambda: torch.ones(10), "big"), "small")
    """

    with _cache_cls_to_use(cache_name, overwrite=False) as cache:
        if not cache.exists:
            cache.v = compute_function(*args, **kw)
    return cache.v


def compute_and_maybe_cache(compute_function, cache_name, *args, _cache_cls_to_use=TorchCache, **kw):
    if cache_name:
        return compute_and_cache(compute_function, cache_name, *args, _cache_cls_to_use=_cache_cls_to_use, **kw)
    else:
        return compute_function(*args, **kw)
