import torch.nn as nn
import torch
from typing import List, Optional, Iterator, Tuple
__all__ = ["traverse_model", "traverse_params_buffs"]


def traverse_model(model: nn.Module, depth: int = 1000, basic_block: Optional[List[nn.Module]] = None, full=False)->Iterator[Tuple[nn.Module, str, nn.Module]]:
    prefix = type(model).__name__
    yield from _traverse_model(model, depth, prefix, basic_block, full)


def _traverse_model(module: nn.Module, depth, prefix, basic_block, full):
    for name, sub_module in module._modules.items():
        scope = prefix+"/"+type(sub_module).__name__+f"[{name}]"
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module
            yield from _traverse_model(sub_module, depth-1, prefix + "/"+type(
                sub_module).__name__+f"[{name}]", basic_block, full)


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
