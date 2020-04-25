from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from itertools import chain

import torch
import torch.nn as nn
from torch import Tensor
from torch._overrides import get_overridable_functions

from models.normal import resnet18
from models.normal.vision_models.ResNet import BasicBlock
from pytorch_Gpipe.utils import traverse_model

# DONE
# create a TracedValue which will record operations
# record all tensor creation ops
# write a model tracer which records current context
# record basic operations on wrapped values
# record layers


# TODO record model weights and biases

# for example given f([tracedValue0,(tracedValue1,2,3),1])
# we need to generate nodes for the constants and to the nested tuple and the wrapping list
# but we need to unwrap the nested values creating edges between them and the container which they reside in

# TODO ensure topological sort which respects chronological order

##############################
# Tracing Metadata
##############################

FUNCTION_NAMESPACE = dict()

CURRENT_SCOPE = ""

##############################
# Graph Metadata
##############################

IN_EDGES = defaultdict(list)
OUT_EDGES = defaultdict(list)

NODE_SCOPES = dict()

KWARGS = defaultdict(dict)
ARGS = defaultdict(list)


##############################
# Tracing Wrappers
##############################


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

    def __call__(self, *args, **kwargs):
        print(f"calling Tensor producing function {self.original_function}")

        # record the operation
        args, kwargs = record_args_and_kwargs(*args, **kwargs)
        out = TracedValue(f"/torch::{self.function_name}")
        record_function_args_and_kwargs(out.id, args, kwargs)

        # perform the operation
        args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)
        out.set_data(self.original_function(*args, **kwargs))

        return out

    def restore_binding(self):
        setattr(self.namespace,
                self.function_name,
                self.original_function)


def delegate_to_traced_value(func):
    @wraps(func)
    def wrapper(*args):
        # record the operation
        args, _ = record_args_and_kwargs(*args)
        traced_self = args[0]
        op_name = func.__name__
        out = TracedValue(f"/{type(traced_self._data)}::{op_name}")
        record_function_args_and_kwargs(out.id, args)

        # retrive the operation implemntation of the traced value
        # and perform it on the inputs
        args, _ = unpack_traced_args_and_kwargs(*args)
        actual_op = getattr(type(args[0]), op_name)
        out.set_data(actual_op(*args))

        return out

    return wrapper


def tracing_not_supported(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        namespace = type(args[0]._data).__name__
        op = func.__name__

        msg = f"tracing {namespace}::{op} is currently not supported"
        raise NotImplementedError(msg)

    return wrapper


class TracedValue(object):
    """
    a wrapper that traces operations done on a value
    for Tensor values we leverage the __torch_function__ API

    functions and attributes are delegated to the wrapped value
    """

    ID = 0

    def __init__(self, creating_op=""):
        self._data = None
        self.namespace = ""
        self.id = TracedValue.ID
        TracedValue.ID += 1

        self.scope = CURRENT_SCOPE + f"{creating_op}"

        # register the traced value
        NODE_SCOPES[self.id] = self.scope
        IN_EDGES[self.id] = []
        OUT_EDGES[self.id] = []
        ARGS[self.id] = []
        KWARGS[self.id] = dict()

    def set_data(self, data):
        assert isTracedValue(
            data), f"TracedValue expects a basic type got {type(data)} scope {self.scope}"
        self._data = data
        self.namespace = f"{type(self._data).__name__}"

    def __repr__(self):
        return f"Node ID:{self.id}\nScope:{self.scope}\nvalue: {self._data}\n"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # operation name
        func_name = func.__name__
        namespace = FUNCTION_NAMESPACE[func].__name__
        op = f"/{namespace}::{func_name}"

        args, kwargs = record_args_and_kwargs(*args, **kwargs)

        # record the operation
        out = TracedValue(op)
        record_function_args_and_kwargs(out.id, args, kwargs)

        # perform the operation
        args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)
        out.set_data(func(*args, **kwargs))

        return out

    def __getattr__(self, name):
        assert isinstance(name,
                          str), f"getattr support only for string args got {type(name)}"

        print(f"accessing attribute {name} of traced value\n")
        out = getattr(self._data, name)
        if isTracedValue(out):
            ret = TracedValue(f"/{self.namespace}::{name}")
            ret.set_data(out)
            record_edge(self.id, ret.id)
            ARGS[ret.id].insert(0, self.id)
            return ret

        return TracedFunction(self.id, self.namespace, out)

    ##############################
    # Magic Method delegation
    ##############################

    @delegate_to_traced_value
    def __getitem__(self, idx):
        pass

    ##############################
    # Arithmetic operations
    ##############################

    @delegate_to_traced_value
    def __add__(self, other):
        pass

    @delegate_to_traced_value
    def __radd__(self, other):
        pass

    @delegate_to_traced_value
    def __iadd__(self, other):
        pass

    @delegate_to_traced_value
    def __sub__(self, other):
        pass

    @delegate_to_traced_value
    def __rsub__(self, other):
        pass

    @delegate_to_traced_value
    def __isub__(self, other):
        pass

    @delegate_to_traced_value
    def __mul__(self, other):
        pass

    @delegate_to_traced_value
    def __rmul__(self, other):
        pass

    @delegate_to_traced_value
    def __imul__(self, other):
        pass

    @delegate_to_traced_value
    def __div__(self, other):
        pass

    @delegate_to_traced_value
    def __rdiv__(self, other):
        pass

    @delegate_to_traced_value
    def __idiv__(self, other):
        pass

    @delegate_to_traced_value
    def __mod__(self, other):
        pass

    @delegate_to_traced_value
    def __matmul__(self, other):
        pass

    @delegate_to_traced_value
    def __pow__(self, other):
        pass

    @delegate_to_traced_value
    def __truediv__(self, other):
        pass

    @delegate_to_traced_value
    def __floordiv__(self, other):
        pass

    @delegate_to_traced_value
    def __rfloordiv__(self, other):
        pass

    @delegate_to_traced_value
    def __rshift__(self, other):
        pass

    @delegate_to_traced_value
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

    def __call__(self, *args, **kwargs):
        print(f"Invoking function {self._func} of wrapped value\n")

        # record the operation
        args, kwargs = record_args_and_kwargs(*args, **kwargs)
        out = TracedValue(f"/{self.namespace}::{self._func.__name__}")
        record_edge(self.self_id, out.id)
        ARGS[out.id].insert(0, self.self_id)
        record_function_args_and_kwargs(out.id, args, kwargs)

        # perform the operation
        args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)
        # NOTE
        # self._func = a.func
        # self._func() is equivalent to a.func() equivalent to type(a).func(a)
        # the a_self is baked in implicitly inside of self._func
        out.set_data(self._func(*args, **kwargs))

        return out


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
        global CURRENT_SCOPE
        CURRENT_SCOPE += f"/{self.name}"
        s = "terminal" if self.terminal else "non terminal"
        print(f"entering {s} {CURRENT_SCOPE}")

        args, kwargs = record_args_and_kwargs(*args, **kwargs)

        if self.terminal:
            # NOTE no need to set the creating operation
            # for terminal layer the layer itself is the creating operation
            out = TracedValue()
            record_function_args_and_kwargs(out.id, args, kwargs)

            args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)
            out.set_data(self.module(*args, **kwargs))
        else:
            out = self.module(*args, **kwargs)

        print(f"leaving {s} {CURRENT_SCOPE}")
        CURRENT_SCOPE = CURRENT_SCOPE.rsplit("/", maxsplit=1)[0]

        assert isinstance(out, TracedValue)

        return out


def isTracedValue(data):
    """
    predicate to check if a value can be traced
    """
    return isinstance(data, (type(None), list, tuple, int, bool, str, float,
                             torch.device, torch.Size, torch.Tensor,
                             torch.dtype, torch.memory_format))


##############################
# Tracing procedure
##############################

def trace(module: nn.Module, args=(), kwargs=None, depth=1000, basic_blocks=()):
    if not isinstance(args, tuple):
        args = (args,)
    if kwargs is None:
        kwargs = dict()

    layers_dict = _wrap_traced_layers(module, depth=depth,
                                      basic_blocks=basic_blocks)
    global CURRENT_SCOPE
    CURRENT_SCOPE = f"{type(module).__name__}"

    with patch_tensor_creating_functions():
        out = module(*args, **kwargs)

    _unwrap_layers(module)

    for m in module.modules():
        assert not isinstance(m, TracedLayer)

    assert CURRENT_SCOPE == f"{type(module).__name__}", CURRENT_SCOPE
    CURRENT_SCOPE = ""


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

    FUNCTION_NAMESPACE.clear()
    # After yield as the exit method
    for f in patched_functions:
        f.restore_binding()


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


##############################
# Graph visualization
##############################


def build_dot():
    theme = {
        "background_color": "#FFFFFF",
        "fill_color": "#E8E8E8",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Times",
        "font_size": "10",
        "margin": "0,0",
        "padding": "1.0,0.5"
    }
    from graphviz import Digraph

    dot = Digraph()
    dot.attr("graph",
             concentrate="true",
             bgcolor=theme["background_color"],
             color=theme["outline_color"],
             fontsize=theme["font_size"],
             fontcolor=theme["font_color"],
             fontname=theme["font_name"],
             margin=theme["margin"],
             rankdir="TB",
             pad=theme["padding"])

    dot.attr("node",
             shape="box",
             style="filled",
             margin="0,0",
             fillcolor=theme["fill_color"],
             color=theme["outline_color"],
             fontsize=theme["font_size"],
             fontcolor=theme["font_color"],
             fontname=theme["font_name"])

    dot.attr("edge",
             style="solid",
             color=theme["outline_color"],
             fontsize=theme["font_size"],
             fontcolor=theme["font_color"],
             fontname=theme["font_name"])

    # add nodes
    for idx, s in NODE_SCOPES.items():
        dot.node(str(idx), s, fillcolor="grey")

    # add edges
    for idx, in_nodes in IN_EDGES.items():
        args, kwargs = ARGS[idx], KWARGS[idx]
        for i in in_nodes:
            if i in kwargs:
                label = f"kwarg: {kwargs[i]}"
            else:
                label = f"arg: {args.index(i)}"
            dot.edge(str(i), str(idx), label=label)
    return dot


def show_graph():
    build_dot().render(filename='graph', directory='.', cleanup=True, format='pdf')

##############################
# recording of function args and kwargs
# support for nested iterables and mix and match of traced and untraced values
##############################


def record_args_and_kwargs(*args, **kwargs):
    """ recording of args and kwargs input format
        this will record all literal values lists/dicts/ints etch
        and build the necessary hierarchy in the graph
        for list/tuple/set elements we record their position
        for dictionaries we record the keywords used

        note that a TracedValue cannot be a dictionary key
    """
    recorded_args = record_args(args, top_level=True, parent_id=-1)
    recorded_kwargs = record_kwargs(kwargs, top_level=True, parent_id=-1)

    return recorded_args, recorded_kwargs


def record_args(args, top_level=True, parent_id=-1):
    new_args = []
    for a in args:
        if isinstance(a, (list, tuple, set)):
            traced_value = TracedValue(
                "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(record_args(a,
                                                      top_level=False,
                                                      parent_id=traced_value.id)))
        elif isinstance(a, dict):
            traced_value = TracedValue(
                "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(record_kwargs(a,
                                                        top_level=False,
                                                        parent_id=traced_value.id)))
        elif isinstance(a, TracedValue):
            traced_value = a
        else:
            s = "Tensor" if isinstance(a, Tensor) else a
            traced_value = TracedValue(f"/prim::Constant_{s}")
            traced_value.set_data(a)
        if parent_id >= 0:
            record_edge(traced_value.id, parent_id)
            ARGS[parent_id].append(traced_value.id)

        if top_level:
            new_args.append(traced_value)
        else:
            new_args.append(traced_value._data)

    return new_args


def record_kwargs(kwargs, top_level=True, parent_id=-1):
    new_kwargs = dict()
    for k, v in kwargs.items():
        assert isinstance(k, (int, bool, str,
                              float, type(None))), f"unsupported kwargs {type(k)}"
        if isinstance(v, (list, tuple, set)):
            traced_value = TracedValue(
                "/" + container_construct_op_name(type(v)))
            traced_value.set_data(type(v)(record_args(v,
                                                      top_level=False,
                                                      parent_id=traced_value.id)))
        elif isinstance(v, dict):
            traced_value = TracedValue(
                "/" + container_construct_op_name(type(v)))
            traced_value.set_data(type(v)(record_kwargs(v,
                                                        top_level=False,
                                                        parent_id=traced_value.id)))
        elif isinstance(v, TracedValue):
            traced_value = v
        else:
            s = "Tensor" if isinstance(v, Tensor) else v
            traced_value = TracedValue(f"/prim::Constant_{s}")
            traced_value.set_data(v)
        if parent_id >= 0:
            record_edge(traced_value.id, parent_id)
            KWARGS[parent_id][traced_value.id] = k

        if top_level:
            new_kwargs[k] = traced_value
        else:
            new_kwargs[k] = traced_value._data

    return new_kwargs


def unpack_traced_args_and_kwargs(*traced_args, **traced_kwargs):
    args = [a._data for a in traced_args]
    kwargs = {k: v._data for k, v in traced_kwargs.items()}

    return args, kwargs


def record_function_args_and_kwargs(out_id, traced_args, traced_kwargs=None):
    if traced_kwargs is None:
        traced_kwargs = dict()

    for a in traced_args:
        ARGS[out_id].append(a.id)
        record_edge(a.id, out_id)

    for k, v in traced_kwargs.items():
        KWARGS[out_id][v.id] = k
        record_edge(v.id, out_id)


def record_edge(src, dest):
    assert src < dest
    # record the edge
    print(f"\n recording edge {src} => {dest}\n")
    IN_EDGES[dest].append(src)
    OUT_EDGES[src].append(dest)


def container_construct_op_name(container_cls):
    container_str = {dict: "Dict",
                     list: "List",
                     tuple: "Tuple",
                     set: "Set"
                     }[container_cls]

    return f"prim::{container_str}Construct"


##############################
# check that the graph is valid
# ensure all recorded data is consistent
##############################

def check_is_valid_graph():
    valid = True
    n = len(IN_EDGES)
    assert len(IN_EDGES) == len(OUT_EDGES)
    assert len(IN_EDGES) == len(NODE_SCOPES)

    for i in range(n):
        if len(IN_EDGES[i]) != (len(ARGS[i]) + len(KWARGS[i])):
            print("wrong number of in edges")
            print(NODE_SCOPES[i])
            print(IN_EDGES[i])
            print(ARGS[i])
            print(KWARGS[i])
            print()
            valid = False

        if not all(a == b for a, b in zip(IN_EDGES[i], chain(ARGS[i], KWARGS[i].keys()))):
            print("arguments are not in the same order")
            print(NODE_SCOPES[i])
            print(IN_EDGES[i])
            print(ARGS[i])
            print(KWARGS[i])
            print()
            valid = False

        for o in OUT_EDGES[i]:
            if i == o:
                print("self cycle")
                print(NODE_SCOPES[i])
                print(IN_EDGES[i])
                print(ARGS[i])
                print(KWARGS[i])
                print()
                valid = False

            if o < i:
                print("violation of topological sort")
                print(NODE_SCOPES[i])
                print(IN_EDGES[i])
                print(ARGS[i])
                print(KWARGS[i])
                print()
            valid = False

    return valid


if __name__ == "__main__":
    if True:
        with patch_tensor_creating_functions():
            t = torch.randn(10, 10)
            t.view(size=[t.shape[0], 2, 5])
            # d = {"size": [t.shape[0], 2, 5]}
            # t.view(**d)
            # print(record_kwargs(d))

            assert isinstance(torch.randn(10, 10), TracedValue)

            m = torch.as_tensor([[1, 2], [3, 4]])
            assert isinstance(m, TracedValue)
            print(m)

            t = torch.tensor([[1, 2], [1, 2]])
            assert isinstance(t, TracedValue)

            print(t)

            t = torch.add(t, m)
            assert isinstance(t, TracedValue)
            t = m.to(device="cuda")
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
            m.add
            print(c.shape)
            print(c.view(c.size(0), 2, 1, 1, 1, 1).size())

            print(c.sum(dim=0))

            show_graph()

            check_is_valid_graph()

    else:

        m = resnet18()
        t = torch.randn(10, 3, 224, 224)
        m(t)
        trace(m, t, basic_blocks=(BasicBlock,))
        print()

        show_graph()

        check_is_valid_graph()

    for i in range(len(IN_EDGES)):
        print(NODE_SCOPES[i])
