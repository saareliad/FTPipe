from contextlib import contextmanager
from functools import wraps
from itertools import chain
import operator
import inspect
from enum import IntEnum

import torch
import torch.nn as nn
from torch import Tensor
from torch._overrides import get_overridable_functions

from pytorch_Gpipe.utils import traverse_model

##############################
# Tracing Metadata
##############################
FUNCTION_NAMESPACE = dict()

CURRENT_SCOPE = ""

NODES = dict()


##############################
# Node construct
##############################
class NodeTypes(IntEnum):
    '''
    Enum representing the possible types of Nodes in the Graph
    '''
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4
    CONSTANT = 5

    def __repr__(self):
        return self.name


# TODO integrate with original Node
class Node():
    def __init__(self, node_type, idx, scope):
        self.type = node_type
        self.id = idx
        self.scope = scope

        self.stage_id = 0
        self.profie = None

        self.out_edges = set()
        self.args = []
        self.kwargs = dict()
        self.value_type = None

        self.tensor_dtype = None
        self.tensor_shape = None

        self.constant_value = None
        NODES[self.id] = (self)

    def add_kwarg(self, kwarg, kwarg_node):
        self.kwargs[kwarg_node] = kwarg

    def add_arg(self, arg_node):
        self.args.append(arg_node)

    def add_out_edge(self, dest_node):
        self.out_edges.add(dest_node)

    def remove_output(self, out_node):
        self.out_edges.remove(out_node)

    @property
    def in_edges(self):
        return list(chain(self.args, self.kwargs.keys()))


##############################
# Tracing Wrappers
##############################
class TracedFunctions():
    functions = set()

    @classmethod
    def register_function(cls, function, namespace=None):
        if namespace is None:
            namespace = inspect.getmodule(function)
            if namespace is None:
                namespace = function.__module__

        if namespace is None:
            raise ValueError(f"could not resolve module for {function}")

        assert hasattr(namespace, function.__name__)

        traced_function = TracedFunction(namespace, function)

        cls.functions.add(traced_function)

    @classmethod
    def enable_tracing(cls):
        for f in cls.functions:
            f.replace_binding()

    @classmethod
    def disable_tracing(cls):
        for f in cls.functions:
            f.restore_binding()

    @classmethod
    def traced_namespaces(cls):
        return {f.namespace for f in cls.functions}


class TracedFunction():
    """
    a Wrapper of an arbitrary static function
    like torch.zeros or math.sqrt which are not invoked from a traced value
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
        out = TracedValue(NodeTypes.OP,
                          f"/{self.namespace.__name__}::{self.function_name}")
        connect_inputs_to_output(out.id, args, kwargs)

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
        op_name = func.__name__

        # if we have (1,) + tracedValue((2,))
        # func will be __radd__ but tuple does not have an __radd__ method
        # so we convert the right hand op to the corresponding left hand op
        # recording an __add__ instead of __radd__
        if op_name in r_arithmetic_ops:
            op_name = "__" + op_name[3:]
            args = tuple(reversed(args))

        traced_self = args[0]
        print(f"delegating {op_name} to {type(traced_self._data)}")
        out = TracedValue(NodeTypes.OP,
                          f"/{type(traced_self._data).__name__}::{op_name}")
        connect_inputs_to_output(out.id, args)

        args, _ = unpack_traced_args_and_kwargs(*args)

        # invoke the operation
        # NOTE some classes like Tensor, do not support invoking all of the __magic__ methods directly
        # so we first invoke them through the operator module who provided functional bindings to most of the __magic__
        # if operator does not have binding for the invoked function we fall back to invoking it using the class method
        # passing an explicit self argument
        try:
            actual_op = getattr(operator, op_name)
            out.set_data(actual_op(*args))
        except Exception:
            print("exception in delegation of ", op_name)
            actual_op = getattr(type(traced_self._data), op_name)
            out.set_data(actual_op(*args))

        return out

    return wrapper


def tracing_not_supported(func):
    """a decortaor to have pretty error messages when accessing an unsupported
    __magic__ method
    """
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
    for other values we trace all instance methods invoked
    and methods that are patched in trace_registered_functions

    functions and attributes are delegated to the wrapped value
    by delegating the __getattr__ of the wrapped to the __getattribute__ of the value

    __magic__ methods require explicit delegation using @delegate_to_traced_value to mark the delegation
    """

    ID = 0

    def __init__(self, node_type, creating_op):
        self._data = None
        self.namespace = ""
        self.id = TracedValue.ID
        TracedValue.ID += 1

        self.scope = CURRENT_SCOPE + f"{creating_op}"

        self.node = Node(node_type, self.id, self.scope)

    def set_data(self, data):
        assert isTracedValue(
            data), f"TracedValue expects a basic type got {type(data)} scope {self.scope}"
        self._data = data
        self.namespace = f"{type(self._data).__name__}"
        self.node.value_type = type(data)
        if isinstance(data, Tensor):
            self.node.tensor_dtype = data.dtype
            self.node.tensor_shape = data.shape

    def __repr__(self):
        return f"Node ID:{self.id}\nScope:{self.scope}\nvalue: {self._data}\n"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # operation name
        func_name = func.__name__
        namespace = FUNCTION_NAMESPACE[func].__name__
        op = f"/{namespace}::{func_name}"

        # record the operation
        args, kwargs = record_args_and_kwargs(*args, **kwargs)
        out = TracedValue(NodeTypes.OP, op)
        print(f"\ncalling {out.scope}")
        connect_inputs_to_output(out.id, args, kwargs)

        # perform the operation
        args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)
        out.set_data(func(*args, **kwargs))

        return out

    def __getattr__(self, name):
        """handles tracing of accessed attributes of traced values
        """
        assert isinstance(name,
                          str), f"getattr support only for string args got {type(name)}"

        print(f"accessing attribute {name} of traced value\n")
        out = getattr(self._data, name)
        if isTracedValue(out):
            name_arg = TracedValue(NodeTypes.CONSTANT, "/prim::Constant")
            name_arg.set_data(name)
            name_arg.node.constant_value = name

            ret = TracedValue(NodeTypes.OP,
                              f"/{self.namespace}::__getattribute__")
            ret.set_data(out)
            record_arg(ret.id, self.id)
            record_arg(ret.id, name_arg.id)
            return ret

        return TracedInstanceFunction(self.id, self.namespace, out)

    ##############################
    # Magic Method delegation
    ##############################

    @delegate_to_traced_value
    def __getitem__(self, idx):
        pass

    # support for storing TracedValue in sets
    @tracing_not_supported
    def __hash__(self):
        pass

    # support for slice assignment
    @delegate_to_traced_value
    def __setitem__(self, idx, value):
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
    @delegate_to_traced_value
    def __eq__(self, other):
        pass

    @delegate_to_traced_value
    def ne(self, other):
        pass

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
    def __or__(self, other):
        return self.__or__(other)

    @tracing_not_supported
    def __xor__(self, other):
        return self.__xor__(other)


class TracedInstanceFunction(object):
    """when we call a function what happens is obj.__getattribute__(self,func_name)(self,*args,**kwargs)
       TracedInstanceFunction is used to record the call operation and it's output
       obj.__getattribute__(self,func_name) returns a TracedInstanceFunction object
       whose __call__ will record the return value
    """

    def __init__(self, self_id, namespace, func):
        self._func = func
        self.self_id = self_id
        self.namespace = namespace

    def __call__(self, *args, **kwargs):
        print(f"Invoking function {self._func.__name__} of wrapped value\n")

        # record the operation
        args, kwargs = record_args_and_kwargs(*args, **kwargs)
        out = TracedValue(NodeTypes.OP,
                          f"/{self.namespace}::{self._func.__name__}")
        record_arg(out.id, self.self_id)
        connect_inputs_to_output(out.id, args, kwargs)

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
        if CURRENT_SCOPE == "":
            CURRENT_SCOPE = self.name
        else:
            CURRENT_SCOPE += f"/{self.name}"

        s = "terminal" if self.terminal else "non terminal"
        print(f"entering {s} scope {CURRENT_SCOPE}")

        args, kwargs = record_args_and_kwargs(*args, **kwargs)

        if self.terminal:
            # NOTE no need to set the creating operation
            # for terminal layer the layer itself is the creating operation
            out = TracedValue(NodeTypes.LAYER, "")

            disable_function_tracing()

            connect_inputs_to_output(out.id, args, kwargs)
            args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)

            out.set_data(self.module(*args, **kwargs))

            trace_registered_functions()

        else:
            with record_free_floating_parameters_and_buffers(self.module):
                out = self.module(*args, **kwargs)
                out = record_non_terminal_output(out)

        print(f"leaving {s} scope {CURRENT_SCOPE}")
        CURRENT_SCOPE = CURRENT_SCOPE.rsplit("/", maxsplit=1)[0]

        assert isinstance(
            out, TracedValue), f"expected layer output of type TracedValue got {type(out)}"
        return out


def isTracedValue(data):
    """
    predicate to check if a value can be traced
    """
    return isinstance(data, (type(None), type(Ellipsis), list, tuple, dict, set, int, bool, str, float, slice,
                             torch.device, torch.Size, torch.Tensor,
                             torch.dtype, torch.memory_format))


##############################
# Tracing procedure
##############################
def trace(module: nn.Module, args=(), kwargs=None, depth=1000, basic_blocks=()):
    reset_tracing_state()
    args, kwargs = prepare_args_and_kwargs(args=args, kwargs=kwargs)

    layers_dict = _wrap_traced_layers(module, depth=depth,
                                      basic_blocks=basic_blocks)

    trace_registered_functions()
    traced_module = TracedLayer(module,
                                name=f"{type(module).__name__}",
                                terminal=False)
    output = traced_module(*args, **kwargs)
    disable_function_tracing()

    output_id = output.id

    _unwrap_layers(module)

    for m in module.modules():
        assert not isinstance(m, TracedLayer)

    global CURRENT_SCOPE
    assert CURRENT_SCOPE == traced_module.name

    CURRENT_SCOPE = ""

    nodes = discard_unused_nodes(NODES)

    nodes, output_id = set_node_indices(nodes, output_id)
    NODES.clear()

    is_valid, errors = check_is_valid_graph(nodes)
    if not is_valid:
        raise RuntimeError(errors)

    return nodes, output_id


def prepare_args_and_kwargs(args=(), kwargs=None):
    # NOTE we cannot use record_args_and_kwargs
    # as they recursively record nested object
    # but semantically we should only record the top level object
    # for example a list input should not record the individual elements
    # until they are accessed
    # we should only know that a list was passed
    # same with kwargs

    if not isinstance(args, tuple):
        args = (args,)
    if kwargs is None:
        kwargs = dict()

    wrapped_args = []
    for idx, a in enumerate(args):
        v = TracedValue(NodeTypes.IN, f"input{idx}")
        v.set_data(a)
        wrapped_args.append(v)

    wrapped_kwargs = dict()
    for i, (k, a) in enumerate(kwargs.items()):
        v = TracedValue(NodeTypes.IN, f"input{idx+1+i}")
        v.set_data(a)
        wrapped_kwargs[k] = v

    return wrapped_args, wrapped_kwargs


def register_new_traced_function(function, namespace=None):
    TracedFunctions.register_function(function, namespace=namespace)


def register_torch_functions():
    for f, namespace in {
        torch.as_tensor: torch,
        torch.from_numpy: torch,
        torch.tensor: torch,
        torch.align_tensors: torch,
        torch.arange: torch,
        torch.as_strided: torch,
        torch.bartlett_window: torch,
        torch.blackman_window: torch,
        torch.empty: torch,
        torch.empty_strided: torch,
        torch.eye: torch,
        torch.from_file: torch,
        torch.full: torch,
        torch.hamming_window: torch,
        torch.hann_window: torch,
        torch.linspace: torch,
        torch.logspace: torch,
        torch.ones: torch,
        torch.rand: torch,
        torch.randn: torch,
        torch.randint: torch,
        torch.randperm: torch,
        torch.range: torch,
        torch.sparse_coo_tensor: torch,
        torch.zeros: torch,
        torch.cat: torch,
        torch.stack: torch
    }.items():
        register_new_traced_function(f, namespace=namespace)


def trace_registered_functions():
    """enable tracing of functions registered functions
    """

    register_torch_functions()

    TracedFunctions.enable_tracing()

    global FUNCTION_NAMESPACE
    FUNCTION_NAMESPACE = {f: ns for ns, funcs in get_overridable_functions().items()
                          for f in funcs}


def disable_function_tracing():
    """revert the patching done by trace_registered_functions
    """
    FUNCTION_NAMESPACE.clear()
    TracedFunctions.disable_tracing()


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


def reset_tracing_state():
    global CURRENT_SCOPE
    CURRENT_SCOPE = ""
    disable_function_tracing()
    NODES.clear()
    FUNCTION_NAMESPACE.clear()
    TracedValue.ID = 0


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
        nested iterables will first record the elements and than the iterable itself
        aka list elements are created before the list itself
        thus ensuring topological/chronological order of traced values

        note that a TracedValue cannot be a dictionary key
    """
    recorded_args, _ = record_args(args, top_level=True)
    recorded_kwargs, _ = record_kwargs(kwargs, top_level=True)

    return recorded_args, recorded_kwargs


def record_args(args, top_level):
    new_args = []
    new_args_id = []
    for a in args:
        if isinstance(a, (list, tuple, set)):
            traced_children, traced_ids = record_args(a,
                                                      top_level=False)
            traced_value = TracedValue(NodeTypes.OP,
                                       "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(traced_children))

            for id in traced_ids:
                record_arg(traced_value.id, id)

        elif isinstance(a, dict):
            traced_children, traced_ids = record_kwargs(a,
                                                        top_level=False)
            traced_value = TracedValue(NodeTypes.OP,
                                       "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(traced_children))

            for k, id in traced_ids.items():
                record_kwarg(traced_value.id, k, id)
        elif isinstance(a, slice):
            traced_children, traced_ids = record_args((a.start, a.stop, a.step),
                                                      top_level=False)
            traced_value = TracedValue(NodeTypes.OP,
                                       "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(*traced_children))

            for id in traced_ids:
                record_arg(traced_value.id, id)

        elif isinstance(a, TracedValue):
            traced_value = a
        else:
            assert not isinstance(
                a, Tensor), "tensor constants should not happen"
            traced_value = TracedValue(NodeTypes.CONSTANT, f"/prim::Constant")
            traced_value.set_data(a)
            traced_value.node.constant_value = a

        if top_level:
            new_args.append(traced_value)
        else:
            new_args.append(traced_value._data)

        new_args_id.append(traced_value.id)

    return new_args, new_args_id


def record_kwargs(kwargs, top_level):
    new_kwargs = dict()
    new_kwargs_ids = dict()
    for k, v in kwargs.items():
        assert isinstance(k, (int, bool, str,
                              float, type(None))), f"unsupported kwargs {type(k)}"
        if isinstance(v, (list, tuple, set)):
            traced_children, children_ids = record_args(v,
                                                        top_level=False)
            traced_value = TracedValue(NodeTypes.OP,
                                       "/" + container_construct_op_name(type(v)))
            traced_value.set_data(type(v)(traced_children))

            for id in children_ids:
                record_arg(traced_value.id, id)

        elif isinstance(v, dict):
            traced_children, traced_ids = record_kwargs(v,
                                                        top_level=False)
            traced_value = TracedValue(NodeTypes.OP,
                                       "/" + container_construct_op_name(type(v)))
            traced_value.set_data(type(v)(traced_children))

            for k, id in traced_ids.items():
                record_kwarg(traced_value.id, k, id)

        elif isinstance(v, TracedValue):
            traced_value = v
        else:
            assert not isinstance(
                v, Tensor), "tensor constants should not happen"
            traced_value = TracedValue(NodeTypes.CONSTANT, f"/prim::Constant")
            traced_value.set_data(v)
            traced_value.node.constant_value = v

        new_kwargs_ids[k] = traced_value.id

        if top_level:
            new_kwargs[k] = traced_value
        else:
            new_kwargs[k] = traced_value._data

    return new_kwargs, new_kwargs_ids


def unpack_traced_args_and_kwargs(*traced_args, **traced_kwargs):
    args = [a._data for a in traced_args]
    kwargs = {k: v._data for k, v in traced_kwargs.items()}

    return args, kwargs


def connect_inputs_to_output(out_id, traced_args, traced_kwargs=None):
    if traced_kwargs is None:
        traced_kwargs = dict()

    for a in traced_args:
        record_arg(out_id, a.id)

    for k, v in traced_kwargs.items():
        record_kwarg(out_id, k, v.id)


@contextmanager
def record_free_floating_parameters_and_buffers(module: nn.Module):
    """
    context manager that records buffers and parameters
    which are not connected to a terminal layer
    """
    for name, t in chain(module.named_parameters(recurse=False), module.named_buffers(recurse=False)):
        traced_t = TracedValue(NodeTypes.BUFF_PARAM,
                               f"/{type(t).__name__}[{name}]")
        traced_t.set_data(t)

        if isinstance(t, nn.Parameter):
            module._parameters[name] = traced_t
        else:
            module._buffers[name] = traced_t

    yield

    # NOTE TracedValue is currently unhashable so we cannot used named_parameters/buffers here
    # so we traverse and modify the state directly
    for name, wrapped_t in chain(module._parameters.items(), module._buffers.items()):
        t = wrapped_t._data
        if isinstance(t, nn.Parameter):
            module._parameters[name] = t
        else:
            module._buffers[name] = t


def record_non_terminal_output(out):
    # NOTE if self.module returns a container
    # tuple/list/set/dict out will be of type container
    # and it will contain traced values but he itself will not be traced
    if not isinstance(out, TracedValue):
        assert isinstance(out,
                          (list, tuple, dict, set)), f"an untraced output of non terminal layer should be a container got {type(out)}"
        traced_out = TracedValue(NodeTypes.OP,
                                 "/" + container_construct_op_name(type(out)))
        if isinstance(out, dict):
            data = dict()
            for k, v in out.items():
                assert isinstance(v,
                                  TracedValue), f"expected a dictionary of traced values got a entry of type {type(v)}"
                record_kwarg(traced_out.id, k, v.id)
                data[k] = v._data
        else:
            data = []
            for v in out:
                assert isinstance(v,
                                  TracedValue), f"expected a container of traced values got a entry of type {type(v)}"
                record_arg(traced_out.id, v.id)
                data.append(v._data)

        traced_out.set_data(type(out)(data))
        return traced_out
    return out


def record_kwarg(node_id, kwarg, kwarg_id):
    assert kwarg_id < node_id
    # record the edge
    print(f"\n recording edge {kwarg_id} => {node_id}\n")
    NODES[kwarg_id].add_out_edge(NODES[node_id])
    NODES[node_id].add_kwarg(kwarg, NODES[kwarg_id])


def record_arg(node_id, arg_id):
    assert arg_id < node_id
    # record the edge
    print(f"\n recording edge {arg_id} => {node_id}\n")
    NODES[arg_id].add_out_edge(NODES[node_id])
    NODES[node_id].add_arg(NODES[arg_id])


def container_construct_op_name(container_cls):
    container_str = {dict: "Dict",
                     list: "List",
                     tuple: "Tuple",
                     set: "Set",
                     slice: "Slice"}[container_cls]

    return f"prim::{container_str}Construct"


##############################
# Graph visualization
##############################
def show_graph(nodes, output_id, filename="graph"):
    build_dot(nodes, output_id).render(filename=filename,
                                       directory='.', cleanup=True, format='pdf')


def build_dot(nodes, output_id):
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
    for node_id in range(len(nodes)):
        node = nodes[node_id]
        scope = node.scope
        value_type = node.value_type
        node_label = f"{scope}\nidx: {node_id}\nvalue type: {value_type}"
        if node_id == output_id:
            node_label += "\nmodel output"
        if node.type is NodeTypes.IN:
            node_label += "\nmodel input"
        if node.type is NodeTypes.CONSTANT:
            node_label += f"\nvalue: {node.constant_value}"

        if issubclass(node.value_type, Tensor):
            node_label += f"\ntensor of type: {node.tensor_dtype}\nshape: {node.tensor_shape}"

        dot.node(str(node_id), label=node_label, fillcolor="grey")

        # add edges
        args, kwargs = node.args, node.kwargs
        for idx, i in enumerate(args):
            dot.edge(str(i.id), str(node_id), label=f"arg: {idx}")
        for i, kw in kwargs.items():
            dot.edge(str(i.id), str(node_id), label=f"kwarg: {kw}")

    return dot


##############################
# check that the graph is valid
# ensure all recorded data is consistent
##############################
def check_is_valid_graph(nodes):
    valid = True

    errors = []
    for i, node in nodes.items():

        if node.type in [NodeTypes.CONSTANT, NodeTypes.IN, NodeTypes.BUFF_PARAM] and len(node.in_edges):
            errors.extend(["leaf node with incoming edges",
                           f"node id: {i}",
                           f"node type: {node.type.__name__}"
                           f"scope: {node.scope}",
                           f"incoming edges: {[n.id for n in node.in_edges]}",
                           f"positional args: {node.args}",
                           f"keyword args: {node.kwargs}",
                           f"outgoing edges: {[n.id for n in node.out_edges]}",
                           ""])
            valid = False

        for o in node.out_edges:
            if i == o.id:
                errors.extend(["self cycle",
                               f"node id: {i}",
                               f"node type: {node.type.__name__}"
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {node.args}",
                               f"keyword args: {node.kwargs}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

            if o.id < i:
                errors.extend(["violation of topological sort",
                               f"node id: {i}",
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {node.args}",
                               f"keyword args: {node.kwargs}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

            if node not in o.in_edges:
                errors.extend(["graph violating back edge not set",
                               f"src id: {i}",
                               f"dest id: {o}",
                               f"src_scope: {node.scope}",
                               f"dest_scope: {o.scope}",
                               f"src_out_edges: {[n.id for n in node.out_edges]}",
                               f"dest_in_edges: {[n.id for n in o.in_edges]}",
                               ""])

                valid = False

        for in_node in node.in_edges:
            if i == in_node.id:
                errors.extend(["self cycle",
                               f"node id: {i}",
                               f"node type: {node.type.__name__}"
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {node.args}",
                               f"keyword args: {node.kwargs}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

            if i < in_node.id:
                errors.extend(["violation of topological sort",
                               f"node id: {i}",
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {node.args}",
                               f"keyword args: {node.kwargs}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

            if node not in in_node.out_edges:
                errors.extend(["graph violating forward edge not set",
                               f"src id: {in_node}",
                               f"dest id: {i}",
                               f"src_scope: {in_node.scope}",
                               f"dest_scope: {node.scope}",
                               f"src_out_edges: {in_node.out_edges}",
                               f"dest_in_edges: {[n.id for n in node.in_edges]}",
                               ""])

                valid = False

        if (node.type != NodeTypes.CONSTANT) and (node.constant_value != None):
            errors.extend(["non constant node with constant_value set",
                           f"node id: {i}",
                           f"scope: {node.scope}",
                           f"value: {node.constant_value}",
                           f"incoming edges: {[n.id for n in node.in_edges]}",
                           f"positional args: {node.args}",
                           f"keyword args: {node.kwargs}",
                           f"outgoing edges: {[n.id for n in node.out_edges]}",
                           ""])
            valid = False

        if node.tensor_shape or node.tensor_dtype or issubclass(node.value_type, Tensor):
            if not ((node.tensor_shape) and (node.tensor_dtype) and (issubclass(node.value_type, Tensor))):
                errors.extend(["tensor value value not recorded in all of TENSOR_SHAPES TENSOR_DTYPES VALUE_TYPES",
                               f"node id: {i}",
                               f"node id: {i}",
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {node.args}",
                               f"keyword args: {node.kwargs}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

    return valid, "\n".join(errors)


##############################
# code generation
##############################
arithmetic_ops = {"__add__": "+",
                  "__sub__": "-",
                  "__mul__": "*",
                  "__div__": "/",
                  "__truediv__": "/",
                  "__floordiv__": "//",
                  "__mod__": "%",
                  "__matmul__": "@",
                  "__pow__": "**"
                  }

r_arithmetic_ops = {"__radd__": "+",
                    "__rsub__": "-",
                    "__rmul__": "*",
                    "__rdiv__": "/",
                    "__rtruediv__": "/",
                    "__rfloordiv__": "//",
                    "__rmod__": "%",
                    "__rmatmul__": "@",
                    "__rpow__": "**"
                    }

inplace_arithmetic_ops = {"__iadd__": " +=",
                          "__isub__": "-=",
                          "__imul__": "*=",
                          "__idiv__": "/=",
                          "__itruediv__": "/=",
                          "__ifloordiv__": "//=",
                          "__imod__": "%=",
                          "__imatmul__": "@=",
                          "__ipow__": "**="}


def compile_model(nodes, output_id, output_file=None):
    statements = []
    ready_expressions = dict()

    model_args = []
    model_kwargs = []
    namespaces = {namespace.__name__ for namespace in
                  chain(get_overridable_functions().keys(), TracedFunctions.traced_namespaces())}

    for idx in range(len(nodes)):
        node = nodes[idx]
        scope = node.scope
        node_type = node.type
        if node_type is NodeTypes.IN:
            # kwarg
            if ":" in scope:
                kw = scope.split(":")[1][1:]
                model_kwargs.append(f"{kw}={node.constant_value}")
                ready_expressions[node] = kw
            else:
                # arg
                model_args.append(scope)
                ready_expressions[node] = scope

        elif node_type is NodeTypes.BUFF_PARAM:
            if node.value_type is Tensor:
                print(f"{idx} buffer")
                ready_expressions[node] = f"self.b_{idx}"
            else:
                print(f"{idx} parameter")
                ready_expressions[node] = f"self.p_{idx}"

        elif node_type is NodeTypes.LAYER:
            print(f"{idx} layer")
            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)

            statements.append(f"l_{idx} = self.l_{idx}({parameter_list})")
            ready_expressions[node] = f"l_{idx}"

        elif node_type is NodeTypes.CONSTANT:
            print(f"{idx} constant")
            ready_expressions[node] = str(node.constant_value)

        elif "prim::DictConstruct" in scope:
            print(f"{idx} dict")
            kwargs = ", ".join([f"'{k}':{ready_expressions[a]}"
                                for a, k in node.kwargs.items()])
            statements.append(f"dict_{idx} = {{{kwargs}}}")
            ready_expressions[node] = f"dict_{idx}"
        elif "prim::SetConstruct" in scope:
            print(f"{idx} set")
            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)
            statements.append(f"set_{idx} = {{{parameter_list}}}")
            ready_expressions[node] = f"set_{idx}"
        elif "prim::ListConstruct" in scope:
            print(f"{idx} list")
            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)
            statements.append(f"list_{idx} = [{parameter_list}]")
            ready_expressions[node] = f"list_{idx}"
        elif "prim::TupleConstruct" in scope:
            print(f"{idx} tuple")
            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)
            if len(node.args) == 1:
                parameter_list += ","
            statements.append(f"tuple_{idx} = ({parameter_list})")
            ready_expressions[node] = f"tuple_{idx}"
        elif "prim::SliceConstruct" in scope:
            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)
            statements.append(f"slice_{idx} = slice({parameter_list})")
            ready_expressions[node] = f"slice_{idx}"
        else:
            # op
            print(f"{idx} op")
            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)
            op_path = scope.rsplit("/", maxsplit=1)[1]
            namespace, func_name = op_path.split("::")

            # function call
            if namespace in namespaces:
                print(f"function call ", func_name)
                statements.append(
                    f"t_{idx} = {namespace}.{func_name}({parameter_list})")
            else:
                param_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions,
                                                     string=False)
                self_arg = param_list[0]
                if "__" not in func_name:
                    print("calling instance method ", func_name)
                    statements.append(
                        f"t_{idx} = {self_arg}.{func_name}({', '.join(param_list[1:])})")
                elif func_name == "__getattribute__":
                    statements.append(f"t_{idx} = {self_arg}.{param_list[1]}")
                elif func_name == "__getitem__":
                    statements.append(f"t_{idx} = {self_arg}[{param_list[1]}]")
                elif func_name == "__setitem__":
                    statements.extend([f"{self_arg}[{param_list[1]}] = {param_list[2]}",
                                       f"t_{idx} = {self_arg}"])
                elif func_name in arithmetic_ops:
                    statements.append(
                        f"t_{idx} = {self_arg} {arithmetic_ops[func_name]} {param_list[1]}")
                elif func_name in inplace_arithmetic_ops:
                    statements.extend(
                        [f"{self_arg} {inplace_arithmetic_ops[func_name]} {param_list[1]}",
                         f"t_{idx} = {self_arg}"])
                elif func_name in r_arithmetic_ops:
                    statements.append(
                        f"t_{idx} = {param_list[1]} {r_arithmetic_ops[func_name]} {self_arg}")
                else:
                    print("calling magic ", func_name)
                    statements.append(
                        f"t_{idx} = {self_arg}.{func_name}({', '.join(param_list[1:])})")
            ready_expressions[node] = f"t_{idx}"

    statements.append(f"return {ready_expressions[nodes[output_id]]}")
    print("\n")
    stage_input_str = ", ".join(["self"] + model_args + model_kwargs)

    forward_decl = f"def forward({stage_input_str}):\n"
    imports = [f"import {namespace}" for namespace in namespaces]

    if output_file is None:
        output_file = f"generated"

    if not output_file.endswith(".py"):
        output_file += ".py"

    with open(output_file, "w") as f:
        f.write("\n".join(imports) + "\n\n")
        f.write(forward_decl)
        f.write("    " + "\n    ".join(statements))


def generate_parameter_list(node_args, node_kwargs, ready_expressions, string=True):
    args = [ready_expressions[a] for a in node_args]
    kwargs = [f"{k}={ready_expressions[a]}"
              for a, k in node_kwargs.items()]
    if string:
        return ", ".join(args + kwargs)
    return args + kwargs


def discard_unused_nodes(nodes):
    new_nodes = []

    for node_id in reversed(range(len(nodes))):
        node = nodes[node_id]
        if node.value_type is None or (node.type is NodeTypes.CONSTANT and (len(node.out_edges) == 0)):
            # a,b=f() will actually invoke __getitem__ 3 times so we discard the last node
            # also discar unused constants
            assert len(
                node.out_edges) == 0, "unused traced value should not have outgoing edges"

            for u in node.in_edges:
                u.remove_output(node)
        else:
            new_nodes.append((node.id, node))

    # reverse dict_order

    return dict(reversed(new_nodes))


def set_node_indices(nodes, output_id):
    if len(nodes) == TracedValue.ID:
        # no nodes were discarded
        return nodes

    new_nodes = dict()

    # populate lookup table for new ids
    for idx, node in enumerate(nodes.values()):
        assert idx <= node.id

        node.id = idx
        new_nodes[idx] = node

    return new_nodes, nodes[output_id].id


##############################
# examples
##############################
class OptionalLayer(nn.Module):
    def __init__(self, func, args=(), kwargs=None):
        super(OptionalLayer, self).__init__()
        self.func = func

        if not isinstance(args, tuple):
            args = (args,)
        if kwargs is None:
            kwargs = dict()

        self.args = args
        self.kwargs = kwargs

    def forward(self, x=None):
        if x is None:
            x = self.func(*self.args, **self.kwargs)

        return x * 10, x + 1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mask_generator = OptionalLayer(torch.zeros, (10, 100))

    def forward(self, x, mask=None):
        # mask = self.mask_generator(mask)

        # o0 = x + mask
        # o1 = x * 2
        # dict_out = {"a": o0, "b": o1}
        # list_out = [o0, o1]
        # tuple_out = (o0, o1)
        # # set_out = {o0, o1}
        # return dict_out

        a, b = self.mask_generator(mask)
        a.device
        b.size()

        t = torch.randn(10, 10, 10, 10)
        ns = t.size(2) // 5
        nd = t.size(2)
        t[:, :, ns - nd:ns, :ns]
        # print(type(t), type(t._data), type(t._data[0]), type(t._data[1]))
        return a, b + x


def test_cases():
    from models.normal import resnet50
    from models.normal.vision_models.ResNet import BasicBlock
    if False:
        trace_registered_functions()
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
        assert isinstance(m.t, TracedInstanceFunction)
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

        disable_function_tracing()
    else:
        import math
        register_new_traced_function(math.sqrt)
        m = resnet50().cuda()
        t = torch.randn(10, 3, 224, 224).cuda()
        # t = torch.randn(10, 100)
        # m = Model()
        mask = None
        # sys.exit()
        # m(x=t)
        args = (t,)
        # kwargs = {"mask": mask}
        kwargs = dict()
        for d in range(3):
            print(f"depth_{d}")
            nodes, output_id = trace(m, depth=d, args=args, kwargs=kwargs)
            print()
            compile_model(nodes, output_id,
                          output_file=f"{type(m).__name__}_depth_{d}")
            show_graph(nodes, output_id,
                       filename=f"{type(m).__name__}_depth_{d}")


if __name__ == "__main__":
    test_cases()
