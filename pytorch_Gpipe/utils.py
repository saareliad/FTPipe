from typing import Iterable, Iterator, List, Optional,\
    Tuple, Union, TypeVar, Generic, OrderedDict, Dict
import collections
import torch
import torch.nn as nn
from torch import Tensor
from itertools import chain

__all__ = ["traverse_model", "traverse_params_buffs",
           "layerDict", "tensorDict"]


def traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None,
                   basic_blocks: Tuple[nn.Module] = (), full: bool = False) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
    '''
    iterate over model layers yielding the layer,layer_scope,encasing_module
    Parameters:
    -----------
    model:
        the model to iterate over
    depth:
        how far down in the model tree to go
    basic_blocks:
        a list of modules that if encountered will not be broken down
    full:
        whether to yield only layers specified by the depth and basick_block options or to yield all layers
    '''
    if prefix is None:
        prefix = type(module).__name__

    for name, sub_module in module.named_children():
        scope = prefix + "/" + type(sub_module).__name__ + f"[{name}]"
        if len(list(sub_module.children())) == 0 or isinstance(sub_module, tuple(basic_blocks)) or depth == 0:
            if full:
                yield sub_module, scope, module, True
            else:
                yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module, False
            yield from traverse_model(sub_module, depth - 1, scope, basic_blocks, full)


def traverse_params_buffs(module: nn.Module, prefix: Optional[str] = None) -> Iterator[Tuple[torch.tensor, str]]:
    '''
    iterate over model's buffers and parameters yielding obj,obj_scope

    Parameters:
    -----------
    model:
        the model to iterate over
    '''
    if prefix is None:
        prefix = type(module).__name__

    # params
    for param_name, param in module.named_parameters(recurse=False):
        param_scope = f"{prefix}/{type(param).__name__}[{param_name}]"
        yield param, param_scope

    # buffs
    for buffer_name, buffer in module.named_buffers(recurse=False):
        buffer_scope = f"{prefix}/{type(buffer).__name__}[{buffer_name}]"
        yield buffer, buffer_scope

    # recurse
    for name, sub_module in module.named_children():
        yield from traverse_params_buffs(sub_module, prefix + "/" + type(sub_module).__name__ + f"[{name}]")


def layerDict(model: nn.Module, depth=1000, basic_blocks=None) -> Dict[str, nn.Module]:
    return {s: l for l, s, _ in traverse_model(model, depth, basic_blocks=basic_blocks)}


def tensorDict(model: nn.Module) -> OrderedDict[str, Tensor]:
    return collections.OrderedDict((s, t)for t, s in traverse_params_buffs(model))


def nested_map(func, ts):
    if isinstance(ts, (list, tuple, set)):
        return type(ts)(nested_map(func, t) for t in ts)
    elif isinstance(ts, dict):
        return {k: nested_map(func, v) for k, v in ts.items()}
    elif isinstance(ts, slice):
        start = nested_map(func, ts.start)
        stop = nested_map(func, ts.stop)
        step = nested_map(func, ts.step)
        return slice(start, stop, step)
    return func(ts)


def flatten(ts):
    if isinstance(ts, (list, tuple, set)):
        yield from chain(*[flatten(t) for t in ts])
    elif isinstance(ts, dict):
        yield from chain(*[flatten(t) for t in ts.values()])
    elif isinstance(ts, slice):
        yield from flatten(ts.start)
        yield from flatten(ts.stop)
        yield from flatten(ts.step)
    else:
        yield ts


def detach_tensors(ts):
    def detach_if_tensor(t):
        if isinstance(t, Tensor):
            # NOTE: it is required for shared stateless!
            # to Set requires grad like the tensor.
            # especially if isinstance(x, torch.nn.Parameter)
            return t.detach().requires_grad_(isinstance(t, torch.nn.Parameter))
        return t

    return nested_map(detach_if_tensor, ts)


def get_tensor_dtypes(ts):
    def get_dtype(t):
        if isinstance(t, Tensor):
            return t.dtype
        return None

    return nested_map(get_dtype, ts)


def get_tensor_shapes(ts):
    def get_shape(t):
        if isinstance(t, Tensor):
            return t.shape
        return None

    return nested_map(get_shape, ts)


def get_device(ts) -> torch.device:
    for t in flatten(ts):
        if isinstance(t, Tensor):
            return t.device

    # default device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
