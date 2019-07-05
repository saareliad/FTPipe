import torch.nn as nn
import torch
from typing import List, Optional, Iterator, Tuple
__all__ = ["traverse_model", "traverse_params_buffs",
           "find_output_shapes_of_scopes", "model_scopes"]


def traverse_model(model: nn.Module, depth: int = 1000, basic_block: Optional[List[nn.Module]] = None, full=False)->Iterator[Tuple[nn.Module, str, nn.Module]]:
    prefix = type(model).__name__
    yield from _traverse_model(model, depth, prefix, basic_block, full)


def _traverse_model(module: nn.Module, depth, prefix, basic_block, full):
    for name, sub_module in module._modules.items():
        scope = prefix+"/"+type(sub_module).__name__+f"[{name}]"
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, tuple(basic_block))) or depth == 0:
            yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module
            yield from _traverse_model(sub_module, depth-1, prefix + "/"+type(
                sub_module).__name__+f"[{name}]", basic_block, full)


def model_scopes(model: nn.Module, depth: int = 1000, basic_block: Optional[List[nn.Module]] = None, full=False)->List[str]:
    return list(map(lambda t: t[1], traverse_model(model, depth=depth, basic_block=basic_block, full=full)))


def traverse_params_buffs(module: nn.Module)->Iterator[Tuple[torch.tensor, str]]:
    prefix = type(module).__name__
    yield from _traverse_params_buffs(module, prefix)


def _traverse_params_buffs(module: nn.Module, prefix):
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


def find_output_shapes_of_scopes(model, scopes, *inputs):
    backup = dict()

    for layer, scope, parent in traverse_model(model, full=True):
        if scope in scopes:
            name = scope[scope.rfind('[')+1:-1]
            parent._modules[name] = ShapeWrapper(layer)

            new_scope = scope[:scope.rfind('/')+1]+f"ShapeWrapper[{name}]"
            backup[new_scope] = (scope, name)

    with torch.no_grad():
        model(*inputs)

    scope_to_shape = {}
    for layer, scope, parent in traverse_model(model, full=True):
        if isinstance(layer, ShapeWrapper):
            old_scope, name = backup[scope]
            scope_to_shape[old_scope] = layer.output_shape
            parent._modules[name] = layer.sub_layer

    return scope_to_shape


class ShapeWrapper(nn.Module):
    def __init__(self, sub_module: nn.Module):
        super(ShapeWrapper, self).__init__()
        self.output_shape = []
        self.sub_layer = sub_module

    def forward(self, *inputs):
        outs = self.sub_layer(*inputs)

        if isinstance(outs, torch.Tensor):
            self.output_shape.append(outs.shape[1:])
        else:
            for t in outs:
                self.output_shape.append(t.shape[1:])
        return outs
