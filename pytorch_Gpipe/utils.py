import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Optional, Iterator, Iterable, Tuple, Union, Dict
__all__ = ["traverse_model", "traverse_params_buffs",
           "find_output_shapes_of_scopes", "model_scopes", "get_device", "_detach_inputs", "_get_size", "_get_shape",
           "Tensors", "TensorsShape", "Devices"]

# the officially supported input types
Tensors = Union[Tensor, List['Tensors'], Tuple['Tensors', ...]]
TensorsShape = Union[Tuple[int], Tuple['TensorsShape'], List['TensorsShape']]

Device = Union[torch.device, int, str]
Devices = Union[List[Device], Tuple[Device, ...]]


def traverse_model(model: nn.Module, depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None, full: Optional[bool] = False) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
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
    prefix = type(model).__name__
    yield from _traverse_model(model, depth, prefix, basic_blocks, full)


def _traverse_model(module: nn.Module, depth: int, prefix: str, basic_blocks: Optional[Iterable[nn.Module]], full: bool) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
    for name, sub_module in module._modules.items():
        scope = prefix+"/"+type(sub_module).__name__+f"[{name}]"
        if len(list(sub_module.children())) == 0 or (basic_blocks != None and isinstance(sub_module, tuple(basic_blocks))) or depth == 0:
            yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module
            yield from _traverse_model(sub_module, depth-1, prefix + "/"+type(
                sub_module).__name__+f"[{name}]", basic_blocks, full)


def model_scopes(model: nn.Module, depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None, full=False) -> List[str]:
    '''
    return a list of all model scopes for the given configuration
     Parameters:
    -----------
    model:
        the model to iterate over
    depth:
        how far down in the model tree to go
    basic_blocks:
        a list of modules that if encountered will not be broken down
    full:
        whether to return only scopes specified by the depth and basick_block options or to yield all scopes up to them
    '''
    return list(map(lambda t: t[1], traverse_model(model, depth=depth, basic_blocks=basic_blocks, full=full)))


def traverse_params_buffs(module: nn.Module) -> Iterator[Tuple[torch.tensor, str]]:
    '''
    iterate over model's buffers and parameters yielding obj,obj_scope

    Parameters:
    -----------
    model:
        the model to iterate over
    '''
    prefix = type(module).__name__
    yield from _traverse_params_buffs(module, prefix)


def _traverse_params_buffs(module: nn.Module, prefix: str) -> Iterator[Tuple[torch.tensor, str]]:
    # params
    for param_name, param in module.named_parameters(recurse=False):
        param_scope = f"{prefix}/{type(param).__name__}[{param_name}]"
        yield param, param_scope

    # buffs
    for buffer_name, buffer in module.named_buffers(recurse=False):
        buffer_scope = f"{prefix}/{type(buffer).__name__}[{buffer_name}]"
        yield buffer, buffer_scope

    # recurse
    for name, sub_module in module._modules.items():
        yield from _traverse_params_buffs(sub_module, prefix + "/"+type(sub_module).__name__+f"[{name}]")


def find_output_shapes_of_scopes(model, scopes, *sample_batch: Tensors) -> Dict:
    '''
    returns a dictionary from scope to input/output shapes without the batch dimention
    by performing a forward pass

    Parameters:
    -----------
    model:
        the model to profile
    scopes:
        the scopes we wish to know their output shapes
    sample_batch:
        the model sample_batch that will used in the forward pass
    '''
    backup = dict()

    for layer, scope, parent in traverse_model(model, full=True):
        if scope in scopes:
            name = scope[scope.rfind('[')+1:-1]
            parent._modules[name] = ShapeWrapper(layer)

            new_scope = scope[:scope.rfind('/')+1]+f"ShapeWrapper[{name}]"
            backup[new_scope] = (scope, name)

    with torch.no_grad():
        model(*sample_batch)

    scope_to_shape = {}
    for layer, scope, parent in traverse_model(model, full=True):
        if isinstance(layer, ShapeWrapper):
            old_scope, name = backup[scope]
            scope_to_shape[old_scope] = (layer.input_shape, layer.output_shape)
            parent._modules[name] = layer.sub_layer

    return scope_to_shape


INCORRECT_INPUT_TYPE = '''currently supported input types are torch.Tensor, List,Tuple or combination of them found: '''


class ShapeWrapper(nn.Module):
    '''
    a wrapper that when it performs forward pass it records the underlying layer's output shape without batch dimention
    '''

    def __init__(self, sub_module: nn.Module):
        super(ShapeWrapper, self).__init__()
        self.output_shape = []
        self.sub_layer = sub_module
        self.input_shape = []

    def forward(self, *inputs: Tensors):
        self.input_shape = _get_shape(inputs)

        outs = self.sub_layer(*inputs)

        self.output_shape = _get_shape(outs)

        return outs


def get_device(x: Tensors) -> Device:
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, (list, tuple)):
        return get_device(x[0])
    raise ValueError(INCORRECT_INPUT_TYPE+f"{type(x)} ")


# TODO consolidate

def _detach_inputs(*inputs: Tensors):
    detached = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            detached.append(x.detach())
        elif isinstance(x, (list, tuple)):
            tmp = []
            for a in x:
                tmp.append(_detach_inputs(a))
            if isinstance(x, tuple):
                detached.append(tuple(tmp))
            else:
                detached.append(tmp)
        else:
            raise ValueError(INCORRECT_INPUT_TYPE+f"{type(x)} ")

    return detached[0] if len(detached) == 1 else tuple(detached)


# for example
# ((5,10,5),(5,55,4)) => ((10,5),(55,4))
def _get_shape(*inputs: Tensors):
    shapes = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            shapes.append((x.shape[1:],))
        elif isinstance(x, (list, tuple)):
            tmp = []
            for a in x:
                tmp.append(*_get_shape(a))
            if isinstance(x, tuple):
                shapes.append(tuple(tmp))
            else:
                shapes.append(tmp)
        else:
            raise ValueError(INCORRECT_INPUT_TYPE+f"{type(x)} {x}")

    return tuple(shapes)


def _get_size(*inputs: Tensors) -> int:
    size = 0
    for x in inputs:
        if isinstance(x, torch.Tensor):
            size += x.nelement() * x.element_size()
        elif isinstance(x, (list, tuple)):
            for a in x:
                size += _get_size(a)
        else:
            raise ValueError(INCORRECT_INPUT_TYPE+f"{type(x)} ")
    return size
