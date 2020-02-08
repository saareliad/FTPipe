from typing import Iterable, Iterator, List, Optional,\
    Tuple, Union, TypeVar, Generic, OrderedDict, Dict
import collections
import torch
import torch.nn as nn
from torch import Tensor
__all__ = ["traverse_model", "traverse_params_buffs",
           "get_device", "_detach_inputs", "_get_size",
           "Tensors", "TensorsShape", "Devices", "OrderedSet", "layerDict", "tensorDict"]

# the officially supported input types
Tensors = Union[Tensor, List['Tensors'], Tuple['Tensors', ...]]
TensorsShape = Union[torch.Size, Tuple['TensorsShape'], List['TensorsShape']]

Device = Union[torch.device, int, str]
Devices = Union[List[Device], Tuple[Device, ...]]


def traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None, basic_blocks: Optional[Iterable[nn.Module]] = None, full: bool = False) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
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
        if len(list(sub_module.children())) == 0 or (basic_blocks != None and isinstance(sub_module, tuple(basic_blocks))) or depth == 0:
            yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module
            yield from traverse_model(sub_module, depth - 1, prefix + "/" + type(
                sub_module).__name__ + f"[{name}]", basic_blocks, full)


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


INCORRECT_INPUT_TYPE = '''currently supported input types are torch.Tensor, List,Tuple or combination of them found: '''


def get_device(x: Tensors) -> Device:
    if hasattr(x, "device"):
        return x.device
    elif isinstance(x, (list, tuple)):
        return get_device(x[0])
    raise ValueError(INCORRECT_INPUT_TYPE + f"{type(x)} ")


def _detach_inputs(*inputs: Tensors):
    detached = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            detached.append(x.detach())
        elif isinstance(x, (list, tuple)):
            tmp = []
            for a in x:
                tmp.append(_detach_inputs(a))
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                # TODO ugly hack
                assert len(tmp) == 4
                detached.append(type(x)(*tmp))
            else:
                detached.append(type(x)(tmp))
        elif x is None:
            detached.append(x)
        else:
            raise ValueError(INCORRECT_INPUT_TYPE + f"{type(x)} ")

    return detached[0] if len(detached) == 1 else tuple(detached)


def _get_size(x: Tensors) -> int:
    size = 0
    shapes = []

    if isinstance(x, torch.Tensor):
        return x.nelement() * x.element_size(), x.shape if len(x.shape) else torch.Size([1])
    elif x is None:
        return 1, torch.Size([])
    elif isinstance(x, (list, tuple)):
        for a in x:
            a_size, a_shape = _get_size(a)
            size += a_size
            shapes.append(a_shape)
        return size, tuple(shapes)
    else:
        raise ValueError(INCORRECT_INPUT_TYPE + f"{type(x)} ")


def flatten(x: Tensors):
    if isinstance(x, Tensor) or x is None:
        return[x]
    elif isinstance(x, (list, tuple)):
        ts = []
        for t in x:
            ts.extend(flatten(t))
        return ts
    else:
        raise ValueError(INCORRECT_INPUT_TYPE + f"{type(x)} ")


T = TypeVar('T')


class OrderedSet(collections.MutableSet, Generic[T]):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]
        return self

    def update(self, keys):
        for k in keys:
            self.add(k)
        return self

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev
        return self

    def __iter__(self) -> Iterator[T]:
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self) -> Iterator[T]:
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def difference_update(self, other):
        for k in self:
            if k in other:
                self.discard(k)
        return self

    def union(self, *others):
        res = OrderedSet(self.map.keys())
        for s in others:
            assert isinstance(s, (set, OrderedSet))
            res.update(s)
        return res

    def difference(self, *others):
        res = OrderedSet()
        for k in self:
            if not any(k in s for s in others):
                res.add(k)
        return res

    def __repr__(self) -> str:
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

    def indexOf(self, key) -> int:
        for idx, k in enumerate(self):
            if key == k:
                return idx
        return -1

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError(f"expected int index got {type(idx).__name__}")
        if idx < 0 or idx >= len(self):
            raise ValueError("index out of range")

        for i, v in enumerate(self):
            if i == idx:
                return v

        raise Exception("should never happen")
