import os
import pickle
import warnings

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
            exception_happened = type is not None
            if not exception_happened:
                assert self.v is not None, "You should enter a value"

                # todo: CREATE DIR
                torch.save(self.v, self.cache_name)
            else:
                print("exception_happened")

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

            exception_happened = type is not None
            if not exception_happened:
                assert self.v is not None, "You should enter a value"
                with open(self.cache_name, "wb") as f:
                    pickle.dump(self.v, f)
            else:
                print("exception_happened")


class GraphCache:
    def __init__(self, cache_name, overwrite=False):
        self.cache_name = cache_name
        self.exists = os.path.exists(cache_name)
        self.overwrite = overwrite
        self.v = None
        self.compute_anyway=False

    def __enter__(self):
        if self.exists:
            try:
                print(f"loading from cache: {self.cache_name}")
                self.v = Graph.deserialize(self.cache_name)
                return self
            except Exception as e:
                self.compute_anyway = True
                warnings.warn("loading from cache failed, (check its consistency!). Will compute value. "
                              f"overwrite={self.overwrite}")

        print(f"computing value for {self.cache_name}")
        return self

    def __exit__(self, type, value, traceback):
        if not self.exists or self.overwrite:
            print(f"saving to cache: {self.cache_name}")
            exception_happened = type is not None
            if not exception_happened:
                assert self.v is not None, "You should enter a value"
                assert isinstance(self.v, Graph)
                self.v.serialize(self.cache_name)
            else:
                print("exception_happened")
        assert isinstance(self.v, Graph)


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
        if not cache.exists or getattr(cache, "compute_anyway", False):
            cache.v = compute_function(*args, **kw)
    return cache.v


def compute_and_maybe_cache(compute_function, cache_name, *args, _cache_cls_to_use=TorchCache, **kw):
    if cache_name:
        return compute_and_cache(compute_function, cache_name, *args, _cache_cls_to_use=_cache_cls_to_use, **kw)
    else:
        return compute_function(*args, **kw)
