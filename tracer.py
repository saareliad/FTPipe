import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._overrides import get_overridable_functions, get_ignored_functions
from pytorch_Gpipe.utils import traverse_model
from models.normal import resnet18
from models.normal.vision_models.ResNet import BasicBlock
from contextlib import contextmanager
from collections import defaultdict
from itertools import chain
import sys


# DONE
# create a TracedValue which will record operations
# record all tensor creation ops
# write a model tracer which records current context
# record basic operations on wrapped values


# TODO record kwargs
# TODO record constants
# TODO record lists and tuples
# TODO record which function was called
# TODO record layers


# NOTE this should be set after we modify tensor creation functions like torch.cat
FUNCTION_NAMESPACE = {}

SCOPE = ""


# basic graph data
IN_EGDES = defaultdict(list)
OUT_EDGES = defaultdict(list)


def record_edge(src, dest):
    # record the edge
    global IN_EGDES
    global OUT_EDGES
    print(f"\n recording edge {src} => {dest}\n")
    IN_EGDES[dest].append(src)
    OUT_EDGES[src].append(dest)


# those function create tensors from potentially non tensor data
# so we wrap them in order to record their creation
TENSOR_PRODUCING_FUNCTIONS = (
    torch.as_tensor,
    torch.from_numpy,
    torch.tensor,
    torch.align_tensors,
    torch.arange,
    torch.as_strided,
    torch.bartlett_window,
    torch.blackman_window,
    torch.empty,
    torch.empty_strided,
    torch.eye,
    torch.from_file,
    torch.full,
    torch.hamming_window,
    torch.hann_window,
    torch.linspace,
    torch.logspace,
    torch.ones,
    torch.rand,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.range,
    torch.sparse_coo_tensor,
    torch.zeros,
    torch.cat,
    torch.stack
)


class TensorProducingFunction():
    """
    a Wrapper of a tensor producing torch function
    like torch.zeros which can produce tensors from untraced values
    the wrapper records the function call and return a TracedValue
    """

    def __init__(self, namespace, original_function):
        self.namespace = namespace
        self.original_function = original_function

    def replace_binding(self):
        setattr(self.namespace, self.original_function.__name__, self)

    def __call__(self, *args, **kwargs):
        print(f"calling Tensor producing function {self.original_function}")
        # TODO record args and kwargs
        u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)
        out = TracedValue(self.original_function(*u_args, **u_kwargs))
        connect_inputs_to_output(args, kwargs, out.id)
        return out

    def restore_binding(self):
        setattr(self.namespace, self.original_function.__name__,
                self.original_function)


@contextmanager
def patch_tensor_functions():
    # Before yield as the enter method
    patched_functions = [TensorProducingFunction(torch, f)
                         for f in TENSOR_PRODUCING_FUNCTIONS]

    for f in patched_functions:
        f.replace_binding()

    global FUNCTION_NAMESPACE
    FUNCTION_NAMESPACE = {f: ns for ns, funcs in get_overridable_functions().items()
                          for f in funcs}
    yield

    # After yield as the exit method
    for f in patched_functions:
        f.restore_binding()


def isTracedValue(data):
    return isinstance(data, (list, tuple, int, bool, str, float,
                             torch.device, torch.Size, torch.Tensor,
                             torch.dtype, torch.memory_format))


def connect_inputs_to_output(args, kwargs, dest_id):
    for a in chain(flatten(args), flatten(kwargs)):
        if isinstance(a, TracedValue):
            record_edge(a.id, dest_id)


def unwrap_args_and_kwargs(*args, **kwargs):
    args = unwrap_args(args)
    kwargs = {k: v._data if isinstance(v, TracedValue) else v
              for k, v in kwargs.items()}
    return args, kwargs


def unwrap_args(args):
    new_args = []
    for t in args:
        if isinstance(t, (list, tuple)):
            new_args.append(type(t)(unwrap_args(t)))
        elif isinstance(t, TracedValue):
            new_args.append(t._data)
        else:
            new_args.append(t)

    return new_args


def flatten(args):
    if isinstance(args, TracedValue):
        yield args
    elif isinstance(args, (list, set, tuple)):
        for a in args:
            yield from flatten(a)
    elif isinstance(args, dict):
        for a in args.values():
            yield from flatten(a)
    else:
        yield args


class TracedValue(object):
    """
    a wrapper that traces operations done on a value
    for Tensor values we leverage the __torch_function__ API

    functions and attributes are delegated to the wrapped value
    """

    ID = 0

    def __init__(self, data, metadata=None):
        assert isTracedValue(
            data), f"TracedValue expects a basic type got {type(data)} scope {SCOPE}"
        self._data = data
        self._metadata = metadata
        self.namespace = f"{type(self._data).__name__}"
        self.id = TracedValue.ID
        TracedValue.ID += 1
        self.scope = SCOPE

    def __repr__(self):
        return "\nID: {}\nProduced in scope: {}\nMetadata:{}\n".format(self.id, self.scope, self._metadata)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # print(f"\ncalled function {func.__name__}")
        # print(FUNCTION_NAMESPACE[func])
        # print(
        #     f"number of args {len(args)}, number of kwargs {len(kwargs)}\n")
        # TODO record args and kwargs
        u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)

        out = func(*u_args, **u_kwargs)
        out = TracedValue(out, metadata=func.__name__)

        connect_inputs_to_output(args, kwargs, out.id)

        return out

    def __getattr__(self, name):
        print(f"accessing attribute {name} of traced value")
        out = getattr(self._data, name)
        print()
        if isTracedValue(out):
            ret = TracedValue(out, metadata=f"{self.namespace}.{name}")
            record_edge(self.id, ret.id)
            return ret

        return TracedFunction(self.id, out)

    def __getitem__(self, idx):
        out = self._data[idx]
        out = TracedValue(out, metadata=f"{self.namespace}.__getitem__({idx})")
        record_edge(self.id, out.id)
        return out
    ##############################
    # Arithmetic operations
    ##############################

    def __add__(self, other):
        out = TracedValue(data=self._data + other,
                          metadata=f"{self.namespace}.add")
        record_edge(self.id, out.id)
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        out = TracedValue(data=self._data * other,
                          metadata=f"{self.namespace}.mul")
        record_edge(self.id, out.id)
        return out

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        if isinstance(other, TracedValue):
            self._data += other._data
        else:
            self._data += other

        # we create a new wrapper that shares the sampe data as the original
        # this is done so we can record the connection between the inputs
        # and the output (we cannot add a self edge)
        out = TracedValue(self._data, metadata=f"{self.namespace}.iadd")
        record_edge(self.id, out.id)
        if isinstance(other, TracedValue):
            record_edge(other.id, out.id)

        return out


class TracedFunction(object):
    """when we call a function of wrapped TracedValue
       we get  TracedValue.__getattr__(func_name).__call__(self,*args,**kwargs)
       TracedFunction is used to record the call operation and it's output
       TracedValue.__getattr__(func_name) returns a TracedFunction object
       whose __call__ will record the return value
    """

    def __init__(self, parent_id, func):
        self._func = func
        self.parent_id = parent_id

    def __call__(self, *args, **kwargs):
        print(f"Invoking function {self._func} of wrapped value")
        # TODO record args and kwargs
        u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)

        # NOTE
        #self._func = a.func
        # self._func() is equivalent to a.func() equivalent to type(a).func(a)
        # the a_self is baked in implicitly inside of self._func
        out = self._func(*u_args, **u_kwargs)

        if isTracedValue(out):
            out = TracedValue(out, metadata=self._func)
            record_edge(self.parent_id, out.id)
            connect_inputs_to_output(args, kwargs, out.id)
            return out
        raise NotImplementedError(
            f"returning function from a function is unsupported got {out}")


def _wrap_traced_layers(module: nn.Module, depth=1000, basic_blocks=()):
    layers_dict = {}
    for sub_layer, scope, parent, terminal in traverse_model(module, depth=depth,
                                                             basic_blocks=basic_blocks,
                                                             full=True):
        name = scope[scope.rfind('[') + 1:-1]

        wrapper = TracedLayer(sub_layer,
                              scope.rsplit('/', maxsplit=1)[1],
                              terminal)
        parent.add_module(name, wrapper)
        layers_dict[scope] = wrapper

    return layers_dict


def _unwrap_layers(module: nn.Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, TracedLayer):
            _unwrap_layers(sub_module.module)
            module.add_module(name, sub_module.module)
        else:
            module.add_module(name, sub_module)


class TracedLayer(nn.Module):
    """ Traced layer is a wrapper around all model layers used for tracing operations
        a traced layer can be terminal as is a layer which will be profiled according to depth and basic  blocks
        and a non terminal layer which is not profiled but still traced

        terminal layers will pass actual non wrapped values to their module
        non terminal will pass wrapped values to their children

    """

    def __init__(self, module: nn.Module, name, terminal):
        super(TracedLayer, self).__init__()
        self.name = name
        self.module = module
        self.terminal = terminal

    def forward(self, *args, **kwargs):
        global SCOPE
        SCOPE += f"/{self.name}"
        s = "terminal" if self.terminal else "non terminal"
        print(f"entering {s} {SCOPE}")

        if self.terminal:
            # TODO record args and kwargs
            u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)
            out = self.module(*u_args, **u_kwargs)
            out = TracedValue(out)
        else:
            out = self.module(*args, **kwargs)

        assert isinstance(out, TracedValue)
        connect_inputs_to_output(args, kwargs, out.id)

        print(f"leaving {s} {SCOPE}")
        SCOPE = SCOPE.rsplit("/", maxsplit=1)[0]

        return out


def trace(module: nn.Module, sample, depth=1000, basic_blocks=()):
    if not isinstance(sample, tuple):
        sample = (sample,)

    layers_dict = _wrap_traced_layers(module, depth=depth,
                                      basic_blocks=basic_blocks)
    global SCOPE
    SCOPE = f"{type(module).__name__}"

    sample = [TracedValue(t) for t in sample]

    with patch_tensor_functions():
        module(*sample)

    SCOPE = ""
    _unwrap_layers(module)

    for m in module.modules():
        assert not isinstance(m, TracedLayer)


if __name__ == "__main__":
    with patch_tensor_functions():
        assert isinstance(torch.randn(10, 10), TracedValue)

        m = torch.as_tensor([[1, 2], [3, 4]])
        assert isinstance(m, TracedValue)

        t = torch.tensor([[1, 2], [1, 2]])
        assert isinstance(t, TracedValue)

        print(t)

        t = torch.add(t, m)
        assert isinstance(t, TracedValue)
        t = m.to("cuda")
        assert isinstance(t, TracedValue)
        print(t)
        # sys.exit()
        assert isinstance(m.device, TracedValue)
        assert isinstance(m.t, TracedFunction)
        a = m * 2
        b = 2 * m

        assert isinstance(t.t().unsqueeze(0).shape[0] + 1, TracedValue)

        m = torch.cat([m, m])
        print(m)

        m + 1
        1 + m
        c = m
        print(m)
        m += 1
        assert c._data is m._data
        print(c)

        print(c.shape)
        print(c.view(c.size(0), 2, 1, 1, 1, 1).size())

        print(c.sum(dim=0))

    m = resnet18()
    t = torch.randn(10, 3, 224, 224)

    trace(m, t, basic_blocks=(BasicBlock,))
