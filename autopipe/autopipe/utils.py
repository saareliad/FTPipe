import inspect
import collections
import operator
import os
import re
from collections import namedtuple
from contextlib import contextmanager
from itertools import chain
from typing import Iterator, Optional, \
    Tuple, OrderedDict, Dict, Type

import torch
import torch.nn as nn
from torch import Tensor


def is_None(a):
    return operator.is_(a, None)


def is_not_None(a):
    return operator.is_not(a, None)


ExecTimes = namedtuple(
    'ExecTimes',
    'forward_time backward_time'
)
FullExecTimes = namedtuple('FullExecTimes', 'recomputation no_recomputation')


# sub_layer, scope, parent, terminal
def traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None,
                   basic_blocks: Tuple[Type[nn.Module]] = (), full: bool = False) -> Iterator[
    Tuple[nn.Module, str, nn.Module, Optional[bool]]]:
    """
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
        whether to yield only layers specified by the depth and basic_block options or to yield all layers
    """
    if prefix is None:
        prefix = type(module).__name__

    for name, sub_module in module.named_children():
        scope = prefix + "/" + type(sub_module).__name__ + f"[{name}]"
        if len(list(sub_module.children())) == 0 or isinstance(sub_module, tuple(basic_blocks)) or depth == 0:
            if full:
                # TODO:
                # is_explicit_block_limit = len(list(sub_module.children())) != 0 and (isinstance(sub_module, tuple(basic_blocks)) or depth == 0)
                yield sub_module, scope, module, True

            else:
                yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module, False
            yield from traverse_model(sub_module, depth - 1, scope, basic_blocks, full)


# sub_layer, scope, parent, terminal
def special_traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None,
                   basic_blocks: Tuple[Type[nn.Module]] = (), special_blocks: Tuple[Type[nn.Module]] = (), next_special_bb_id=None,
                           full: bool = False, mark = False) -> Iterator[
    Tuple[nn.Module, str, nn.Module, Optional[bool], Optional[int]]]:
    """
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
        whether to yield only layers specified by the depth and basic_block options or to yield all layers
    """
    if next_special_bb_id is None:
        next_special_bb_id = 0

    if prefix is None:
        prefix = type(module).__name__

    for child_idx, (name, sub_module) in enumerate(module.named_children()):
        scope = prefix + "/" + type(sub_module).__name__ + f"[{name}]"
        if len(list(sub_module.children())) == 0 or isinstance(sub_module, tuple(basic_blocks)) or depth == 0:
            if full:
                if mark:
                    yield sub_module, scope, module, True, None
                else:
                    yield sub_module, scope, module, True

            else:
                yield sub_module, scope, module
        elif isinstance(sub_module, tuple(special_blocks)):
            # if it is a part of module which is a basic block, it will get parent's mark.

            # TODO: do we want to mark it?
            if mark:
                if not hasattr(module, "_next_special_bb_id"):
                    next_special_bb_id += 0.01
                sub_module._next_special_bb_id = next_special_bb_id

            if full:
                if mark:
                    yield sub_module, scope, module, False, next_special_bb_id
                else:
                    yield sub_module, scope, module, False
            yield from special_traverse_model(sub_module, depth - 1, scope, basic_blocks, special_blocks,
                                              next_special_bb_id+child_idx, full, mark=mark)

        else:
            if full:
                if mark:
                    yield sub_module, scope, module, False, None
                else:
                    yield sub_module, scope, module, False

            yield from special_traverse_model(sub_module, depth - 1, scope, basic_blocks, special_blocks,
                                              next_special_bb_id+child_idx, full, mark=mark)

    # clear the mess
    if mark and hasattr(module, "_next_special_bb_id"):
        delattr(module, "_next_special_bb_id")


def traverse_params_buffs(module: nn.Module, prefix: Optional[str] = None) -> Iterator[Tuple[torch.tensor, str]]:
    """
    iterate over model's buffers and parameters yielding obj,obj_scope

    Parameters:
    -----------
    model:
        the model to iterate over
    """
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


def layerDict(model: nn.Module, depth=1000, basic_blocks=()) -> Dict[str, nn.Module]:
    return {s: l for l, s, _ in traverse_model(model, depth, basic_blocks=basic_blocks)}


def tensorDict(model: nn.Module) -> OrderedDict[str, Tensor]:
    return collections.OrderedDict((s, t) for t, s in traverse_params_buffs(model))


def nested_map(func, ts, full=False):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        return func(ts)
    elif isinstance(ts, (list, tuple, set)):
        return type(ts)(nested_map(func, t, full=full) for t in ts)
    elif isinstance(ts, dict):
        return {k: nested_map(func, v, full=full) for k, v in ts.items()}
    elif isinstance(ts, slice) and full:
        start = nested_map(func, ts.start, full=full)
        stop = nested_map(func, ts.stop, full=full)
        step = nested_map(func, ts.step, full=full)
        return slice(start, stop, step)
    return func(ts)


def flatten(ts):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        yield ts
    elif isinstance(ts, (list, tuple, set)):
        yield from chain(*[flatten(t) for t in ts])
    elif isinstance(ts, dict):
        yield from chain(*[flatten(t) for k, t in sorted(ts.items(), key=lambda t: t[0])])
    else:
        yield ts


def unflatten(xs, structure):
    return _unflatten(xs, structure)[0]


def _unflatten(xs, structure):
    if isinstance(structure, torch.Size):
        # torch.Size is subclass of tuple which is stupid
        return xs[0], 1

    if not isinstance(structure, (list, tuple, set, dict)):
        return xs[0], 1

    if isinstance(structure, (list, tuple, set)):
        offset = 0
        elements = []
        for s in structure:
            e, n = _unflatten(xs[offset:], s)
            elements.append(e)
            offset += n

        return type(structure)(elements), offset

    assert isinstance(structure, dict)
    offset = 0
    elements = dict()
    for k, v in sorted(structure.items(), key=lambda t: t[0]):
        e, n = _unflatten(xs[offset:], v)
        elements[k] = e
        offset += n

    return elements, offset


def detach_tensors(ts):
    def detach_if_tensor(t):
        if isinstance(t, Tensor):
            # NOTE: it is required for shared stateless!
            # to Set requires grad like the tensor.
            # especially if isinstance(x, torch.nn.Parameter)
            return t.detach().requires_grad_(t.requires_grad)
        return t

    return nested_map(detach_if_tensor, ts)


def move_tensors(ts, device):
    def move(t):
        if isinstance(t, (nn.Module, Tensor)):
            return t.to(device)
        return t

    return nested_map(move, ts)


def set_grad_mode(ts, require_grad):
    def grad_mode(t):
        if isinstance(t, Tensor):
            return t.detach().requires_grad_(isinstance(t, nn.Parameter) or (require_grad and t.is_floating_point()))
        return t

    return nested_map(grad_mode, ts)


def get_tensor_dtypes(ts):
    def get_dtype(t):
        if isinstance(t, Tensor):
            return t.dtype
        return type(t)

    return nested_map(get_dtype, ts)


def get_tensor_shapes(ts):
    def get_shape(t):
        if isinstance(t, Tensor):
            return t.shape if t.shape else torch.Size([1])
        elif isinstance(t, torch.Size):
            # HACK send torch.Size in MPI as tuple
            return torch.Size([len(t)])
        return None

    return nested_map(get_shape, ts)


def get_device(ts) -> torch.device:
    for t in flatten(ts):
        if isinstance(t, Tensor):
            return t.device

    # default device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def force_out_of_place(func):
    prev_state = None
    modified = False
    if hasattr(func, "inplace") and isinstance(func.inplace, bool):
        prev_state = func.inplace
        modified = True
        setattr(func, "inplace", False)
    yield

    if modified:
        setattr(func, "inplace", prev_state)

    ##############################
    # Magic Method delegation
    # intentionally explicit
    # NOTE if the method requires specific syntax
    # then it should be also added in model_profiling/tracer.py
    # and ensure correct code generation in compiler/partition_forward_method.generate_magic
    ##############################


arithmetic_ops = {"__add__": "+",
                  "__sub__": "-",
                  "__mul__": "*",
                  "__div__": "/",
                  "__truediv__": "/",
                  "__floordiv__": "//",
                  "__mod__": "%",
                  "__matmul__": "@",
                  "__pow__": "**",
                  "__lshift__": "<<",
                  "__rshift__": ">>",
                  "__and__": "&",
                  "__or__": "|",
                  "__xor__": "^"
                  }

inplace_arithmetic_ops = {f"__i{op_name[2:]}": f"{symbol}=" for op_name, symbol in arithmetic_ops.items()}

r_arithmetic_ops = {f"__r{op_name[2:]}": f"{symbol}" for op_name, symbol in arithmetic_ops.items()}

logical_ops = {"__eq__": "==",
               "__ne__": "!=",
               "__gt__": ">",
               "__ge__": ">=",
               "__lt__": "<",
               "__le__": "<="}

conversion_ops = {"__bool__": "bool"}

unary_ops = {"__neg__": "-",
             "__pos__": "+",
             "__invert__": "~"}

magics = {"__len__": "len",
          "__abs__": "abs",
          "__iter__": "iter"}

# see https://pytorch.org/docs/stable/torch.html#tensor-creation-ops
# track and add

tensor_creation_ops = {
    torch.tensor: torch,
    torch.sparse_coo_tensor: torch,
    torch.as_tensor: torch,
    torch.as_strided: torch,
    torch.from_numpy: torch,
    torch.zeros: torch,
    torch.zeros_like: torch,
    torch.ones: torch,
    torch.ones_like: torch,
    torch.arange: torch,
    torch.range: torch,
    torch.linspace: torch,
    torch.logspace: torch,
    torch.eye: torch,
    torch.empty: torch,
    torch.empty_like: torch,
    torch.empty_strided: torch,
    torch.full: torch,
    torch.full_like: torch,
    torch.quantize_per_tensor: torch,
    torch.quantize_per_channel: torch,
    torch.dequantize: torch,
    torch.complex: torch,
    torch.polar: torch,
    torch.heaviside: torch,
    #
    torch.cat: torch,
    torch.chunk: torch,
    # torch.dsplit: torch,
    # torch.column_stack: torch,
    torch.dstack: torch,
    torch.gather: torch,
    # torch.hsplit: torch,
    torch.hstack: torch,
    torch.index_select: torch,
    torch.masked_select: torch,
    torch.movedim: torch,
    # torch.moveaxis: torch,
    torch.narrow: torch,
    torch.nonzero: torch,
    torch.reshape: torch,
    # torch.row_stack: torch,
    torch.scatter: torch,
    torch.scatter_add: torch,
    torch.split: torch,
    torch.squeeze: torch,
    torch.stack: torch,
    # torch.swapaxes: torch,
    # torch.swapdims: torch,
    torch.t: torch,
    torch.take: torch,
    # torch.take_along_dim: torch,
    # torch.tensor_split: torch,
    # torch.tile: torch,
    torch.transpose: torch,
    torch.unbind: torch,
    torch.unsqueeze: torch,
    # torch.vsplit: torch,
    torch.vstack: torch,
    torch.where: torch,


    torch.align_tensors: torch,
    torch.bartlett_window: torch,
    torch.blackman_window: torch,
    torch.from_file: torch,
    torch.hamming_window: torch,
    torch.hann_window: torch,

    torch.rand: torch,
    torch.randn: torch,
    torch.randint: torch,
    torch.randperm: torch,



    torch.less: torch,
    torch.less_equal: torch,
    torch.lt: torch ## and more!!

}
tensor_creation_ops_without_device_kw = {
    # TODO: maybe more...
    torch.cat: torch,
    torch.stack: torch,
    torch.where: torch
}


def get_call_site(*ignored_files) -> Optional[str]:
    ignored_files = (__file__,) + ignored_files
    curdir = os.path.dirname(os.path.realpath(__file__))
    for f in inspect.stack():
        frameinfo = inspect.getframeinfo(f[0])
        file_name = frameinfo.filename
        if (file_name not in ignored_files) and (not file_name.startswith(curdir)) and (not "torch\\" in file_name):
            return file_name + ", line " + str(frameinfo.lineno) + "\n"

    return None


def convert_none_checks(input_file: str, output_file: str):
    """utility to convert None checks which are unsupported by the traced to
       a convention we support

       we match patters like:
       if identifier is None  => if is_None(identified)
       if identified is not None => if is_not_None(identifier)

    Args:
    ---------------------------------------------------------------
        input_file: str
        path to the python file we wish to convert
        
        output_file:str:
        path to the python output file to which write the result
    """
    res = []
    modified = False
    with open(input_file, 'r') as f:
        for idx, original in enumerate(f.readlines()):
            is_None_pattern = r'([a-zA-Z0-9_\.\(\)\[\]\-\+\*\/]+) is None'
            is_not_None_pattern = r'([a-zA-Z0-9_\.\(\)\[\]\-\+\*\/]+) is not None'
            line = re.sub(is_None_pattern, r'is_None(\1)', original)
            line = re.sub(is_not_None_pattern, r'is_not_None(\1)', line)
            if line != original:
                modified = True
                print(f"-I- changed line {idx}")
                print(f"from {original.lstrip().rstrip()}")
                print(f"to {line.lstrip().rstrip()}")
                print()

            res.append(line)

    if modified:
        lines = ['import operator\n']
        lines.append("\n")
        lines.append("\n")
        lines.append(inspect.getsource(is_None))
        lines.append("\n")
        lines.append("\n")
        lines.append(inspect.getsource(is_not_None))
        lines.append("\n")
        res = lines + res

    with open(output_file, "w") as f:
        f.writelines(res)


