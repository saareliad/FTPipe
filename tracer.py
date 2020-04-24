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
from functools import wraps
import sys
import operator
from collections import deque

# DONE
# create a TracedValue which will record operations
# record all tensor creation ops
# write a model tracer which records current context
# record basic operations on wrapped values
# record layers


# TODO record kwargs
# TODO record constants
# TODO record lists and tuples
# TODO record which function was called


# NOTE this should be set after we modify tensor creation functions like torch.cat
FUNCTION_NAMESPACE = dict()

SCOPE = ""

# basic graph data
IN_EGDES = defaultdict(list)
OUT_EDGES = defaultdict(list)


DEBUG_CREATED_SCOPES = dict()


def record_edge(src, dest):
    assert src < dest
    # record the edge
    global IN_EGDES
    global OUT_EDGES
    print(f"\n recording edge {src} => {dest}\n")
    IN_EGDES[dest].append(src)
    OUT_EDGES[src].append(dest)


def record_edges(args, kwargs, out):
    """connect all edges between traced args and kwargs to the output
    """
    for a in chain(flatten(args), flatten(kwargs)):
        assert isinstance(a, TracedValue)
        record_edge(a.id, out.id)


def connect_inputs_to_output(connect_self=True):
    """decorator to auto add edges between traced input and kwargs of function to it's output
       if connect_self is True will also record the connection between the self of the caller to the output
    """
    def connect(func):
        @wraps(func)
        def wrapper(self_arg, *args, **kwargs):
            out = func(self_arg, *args, **kwargs)

            if connect_self:
                record_edge(self_arg.id, out.id)
            record_edges(args, kwargs, out)

            return out
        return wrapper
    return connect


def ensure_traced_args_and_kwargs(func):
    """a decorator that ensures that all inputs of a function are traced
    """
    @wraps(func)
    def wrapper(self_arg, *args, **kwargs):
        args, kwargs = wrap_args_and_kwargs(*args, **kwargs)
        return func(self_arg, *args, **kwargs)
    return wrapper


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


class TracedTensorProducingFunction():
    """
    a Wrapper of a tensor producing torch function
    like torch.zeros which can produce tensors from untraced values
    the wrapper records the function call and return a TracedValue
    """

    def __init__(self, namespace, original_function):
        self.namespace = namespace
        self.original_function = original_function
        self.function_name = self.original_function.__name__

    def replace_binding(self):
        setattr(self.namespace, self.function_name, self)

    @ensure_traced_args_and_kwargs
    @connect_inputs_to_output(connect_self=False)
    def __call__(self, *args, **kwargs):
        print(f"calling Tensor producing function {self.original_function}")
        # TODO record args and kwargs
        u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)
        out = TracedValue(self.original_function(*u_args, **u_kwargs),
                          f"torch::{self.function_name}")
        return out

    def restore_binding(self):
        setattr(self.namespace,
                self.function_name,
                self.original_function)


@contextmanager
def patch_tensor_creating_functions():
    """context manager that ensures that tensors which are created using torch functions like torch.zeros
    are traced
    """
    # Before yield as the enter method
    patched_functions = [TracedTensorProducingFunction(torch, f)
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
    """
    predicate to check if a value can be traced
    """
    return isinstance(data, (list, tuple, int, bool, str, float,
                             torch.device, torch.Size, torch.Tensor,
                             torch.dtype, torch.memory_format))


def unwrap_args_and_kwargs(*args, **kwargs):
    """
    retrives the traced data from args and kwargs untraced data is returned as is 
    """
    args = unwrap_args(args)
    kwargs = unwrap_kwargs(kwargs)
    return args, kwargs


def wrap_args_and_kwargs(*args, **kwargs):
    """recursively ensure that all arguments passed in args and kwargs are traced
       if a value was not traced it will be recorded as a constant
    """
    args = wrap_args(args)
    kwargs = wrap_kwargs(kwargs)
    return args, kwargs


def wrap_args(args):
    new_args = []
    for t in args:
        if isinstance(t, (list, tuple)):
            new_args.append(type(t)(wrap_args(t)))
        elif isinstance(t, TracedValue):
            new_args.append(t)
        else:
            new_args.append(TracedValue(t, f"prim::constant_{t}"))

    return new_args


def wrap_kwargs(kwargs):
    return{k: v if isinstance(v, TracedValue) else TracedValue(v, f"prim::constant_{v}")
           for k, v in kwargs.items()}


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


def unwrap_kwargs(kwargs):
    return{k: v._data if isinstance(v, TracedValue) else v
           for k, v in kwargs.items()}


def flatten(args):
    """return all nested data and tracedValues from a nested iterable
    """
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


def reduce_inputs(func):
    """
    decorator which ensures that func is called with unwrapped inputs
    """
    @wraps(func)
    def wrapper(*wrapped_args, **wraped_kwargs):
        args, kwargs = unwrap_args_and_kwargs(*wrapped_args, **wraped_kwargs)
        return func(*args, **kwargs)
    return wrapper


def call_traced_fn(func):
    """a decorator which delegates function call to the wrapped value
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        actual_func = getattr(type(args[0]), func_name)
        out = actual_func(*args, **kwargs)
        return out
    return wrapper


def wrap_output(func):
    """a decorator that auto wraps the output of func in a TracedValue
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        namespace = type(args[0]).__name__
        op = func.__name__
        out = func(*args, **kwargs)
        return TracedValue(out, f"{namespace}::{op}")
    return wrapper


def wrap_magic(func):
    """
    a decorator that auto records magic methods

    @wrap_magic
    def __add__(self,other):
        pass

    will ensure to record the add operator as called with the traced value
    """
    return ensure_traced_args_and_kwargs(
        connect_inputs_to_output(connect_self=True)(
            reduce_inputs(
                wrap_output(
                    call_traced_fn(func)
                ))))


def tracing_not_supported(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        namespace = type(args[0]).__name__
        op = func.__name__

        msg = f"tracing {namespace}::{op} is currently not supported"
        raise NotImplementedError(msg)

    return reduce_inputs(wrapper)


class TracedValue(object):
    """
    a wrapper that traces operations done on a value
    for Tensor values we leverage the __torch_function__ API

    functions and attributes are delegated to the wrapped value
    """

    ID = 0

    def __init__(self, data, creating_op):
        assert isTracedValue(
            data), f"TracedValue expects a basic type got {type(data)} scope {SCOPE}"
        self._data = data
        self.namespace = f"{type(self._data).__name__}"
        self.id = TracedValue.ID
        TracedValue.ID += 1

        if creating_op:
            self.scope = SCOPE + f"/{creating_op}"
        else:
            self.scope = SCOPE

        DEBUG_CREATED_SCOPES[self.id] = self.scope

    def __repr__(self):
        return f"Node ID:{self.id}\nScope:{self.scope}\nvalue: {self._data}\n"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        func_name = func.__name__
        namespace = FUNCTION_NAMESPACE[func].__name__
        op = f"/{namespace}::{func_name}"
        # TODO record args and kwargs
        args, kwargs = wrap_args_and_kwargs(*args, **kwargs)
        u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)

        out = func(*u_args, **u_kwargs)
        out = TracedValue(out, creating_op=op)

        record_edges(args, kwargs, out)

        return out

    def __getattr__(self, name):
        print(f"accessing attribute {name} of traced value")
        out = getattr(self._data, name)
        print()
        if isTracedValue(out):
            ret = TracedValue(out, creating_op=f"{self.namespace}::{name}")
            record_edge(self.id, ret.id)
            return ret

        return TracedFunction(self.id, self.namespace, out)

    ##############################
    # Magic Method delegation
    # this is either very clever or a huge abuse of decorators
    ##############################

    @wrap_magic
    def __getitem__(self, idx):
        pass

    ##############################
    # Arithmetic operations
    ##############################
    @wrap_magic
    def __add__(self, other):
        pass

    @wrap_magic
    def __radd__(self, other):
        pass

    @wrap_magic
    def __iadd__(self, other):
        pass

    @wrap_magic
    def __sub__(self, other):
        pass

    @wrap_magic
    def __rsub__(self, other):
        pass

    @wrap_magic
    def __isub__(self, other):
        pass

    @wrap_magic
    def __mul__(self, other):
        pass

    @wrap_magic
    def __rmul__(self, other):
        pass

    @wrap_magic
    def __imul__(self, other):
        pass

    @wrap_magic
    def __div__(self, other):
        pass

    @wrap_magic
    def __rdiv__(self, other):
        pass

    @wrap_magic
    def __idiv__(self, other):
        pass

    @wrap_magic
    def __mod__(self, other):
        pass

    @wrap_magic
    def __matmul__(self, other):
        pass

    @wrap_magic
    def __pow__(self, other):
        pass

    @wrap_magic
    def __truediv__(self, other):
        pass

    @wrap_magic
    def __floordiv__(self, other):
        pass

    @wrap_magic
    def __rfloordiv__(self, other):
        pass

    @wrap_magic
    def __rshift__(self, other):
        pass

    @wrap_magic
    def __lshift__(self, other):
        pass

    ##############################
    # Logical operations
    ##############################
    @tracing_not_supported
    def __and__(self, other):
        return self.__and__(other)

    @tracing_not_supported
    def __ge__(self, other):
        return self.__ge__(other)

    @tracing_not_supported
    def __gt__(self, other):
        return self.__gt__(other)

    @tracing_not_supported
    def __le__(self, other):
        return self.__le__(other)

    @tracing_not_supported
    def __lt__(self, other):
        return self.__lt__(other)

    @tracing_not_supported
    def ne(self, other):
        return self.ne(other)

    @tracing_not_supported
    def __or__(self, other):
        return self.__or__(other)

    @tracing_not_supported
    def __xor__(self, other):
        return self.__xor__(other)

    @tracing_not_supported
    def __eq__(self, other):
        return self.__eq__(other)


class TracedFunction(object):
    """when we call a function of wrapped TracedValue
       we get  TracedValue.__getattr__(func_name).__call__(self,*args,**kwargs)
       TracedFunction is used to record the call operation and it's output
       TracedValue.__getattr__(func_name) returns a TracedFunction object
       whose __call__ will record the return value
    """

    def __init__(self, self_id, namespace, func):
        self._func = func
        self.self_id = self_id
        self.namespace = namespace

    @ensure_traced_args_and_kwargs
    @connect_inputs_to_output(connect_self=False)
    def __call__(self, *args, **kwargs):
        print(f"Invoking function {self._func} of wrapped value")
        # TODO record args and kwargs
        u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)

        # NOTE
        # self._func = a.func
        # self._func() is equivalent to a.func() equivalent to type(a).func(a)
        # the a_self is baked in implicitly inside of self._func
        out = self._func(*u_args, **u_kwargs)

        if isTracedValue(out):
            out = TracedValue(out, f"{self.namespace}::{self._func.__name__}")
            record_edge(self.self_id, out.id)
            return out
        raise NotImplementedError(
            f"returning function from a function is unsupported got {out}")


def _wrap_traced_layers(module: nn.Module, depth=1000, basic_blocks=()):
    layers_dict = dict()
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

    @ensure_traced_args_and_kwargs
    @connect_inputs_to_output(connect_self=False)
    def forward(self, *args, **kwargs):
        global SCOPE
        SCOPE += f"/{self.name}"
        s = "terminal" if self.terminal else "non terminal"
        print(f"entering {s} {SCOPE}")
        if self.terminal:
            # TODO record args and kwargs
            u_args, u_kwargs = unwrap_args_and_kwargs(*args, **kwargs)
            out = self.module(*u_args, **u_kwargs)
            out = TracedValue(out, "")
        else:
            out = self.module(*args, **kwargs)

        print(f"leaving {s} {SCOPE}")
        SCOPE = SCOPE.rsplit("/", maxsplit=1)[0]

        assert isinstance(out, TracedValue)

        return out


def trace(module: nn.Module, sample, depth=1000, basic_blocks=()):
    if not isinstance(sample, tuple):
        sample = (sample,)

    layers_dict = _wrap_traced_layers(module, depth=depth,
                                      basic_blocks=basic_blocks)
    global SCOPE
    SCOPE = f"{type(module).__name__}"

    sample = [TracedValue(t, f"input{idx}") for idx, t in enumerate(sample)]

    with patch_tensor_creating_functions():
        module(*sample)

    _unwrap_layers(module)

    for m in module.modules():
        assert not isinstance(m, TracedLayer)

    assert SCOPE == f"{type(module).__name__}", SCOPE
    SCOPE = ""


if __name__ == "__main__":
    with patch_tensor_creating_functions():
        assert isinstance(torch.randn(10, 10), TracedValue)

        m = torch.as_tensor([[1, 2], [3, 4]])
        assert isinstance(m, TracedValue)
        print(m)

        t = torch.tensor([[1, 2], [1, 2]])
        assert isinstance(t, TracedValue)

        print(t)

        t = torch.add(t, m)
        assert isinstance(t, TracedValue)
        t = m.to("cuda")
        assert isinstance(t, TracedValue)
        print(t)
        assert isinstance(m.device, TracedValue)
        assert isinstance(m.t, TracedFunction)
        m.t()
        a = m * 2
        b = 2 * m
        assert isinstance(a, TracedValue)
        assert isinstance(b, TracedValue)

        s = t.t().unsqueeze(0).shape[0] + 1
        print(s)
        assert isinstance(s, TracedValue)

        m = torch.cat([m, m])
        assert isinstance(m, TracedValue)
        print(m)

        assert isinstance(1 + m, TracedValue)
        assert isinstance(m + 1, TracedValue)
        c = m
        print(m)
        m += 1
        assert c._data is m._data
        print(c)

        print(c.shape)
        print(c.view(c.size(0), 2, 1, 1, 1, 1).size())

        print(c.sum(dim=0))

    print()
    # sys.exit()
    m = resnet18()
    t = torch.randn(10, 3, 224, 224)

    trace(m, t, basic_blocks=(BasicBlock,))
    print()
    print(f"total created scopes: {len(DEBUG_CREATED_SCOPES)}")
    for s in DEBUG_CREATED_SCOPES.values():
        print(s)
