from contextlib import contextmanager
from functools import wraps
from itertools import chain
import operator
import warnings


import torch
import torch.nn as nn
from torch import Tensor
from torch._overrides import get_overridable_functions

from pytorch_Gpipe.utils import traverse_model
from .control_flow_graph import Node, NodeTypes, Graph
from ..utils import get_tensor_shapes, get_tensor_dtypes, r_arithmetic_ops,logical_ops, nested_map
##############################
# Tracing Metadata
##############################
FUNCTION_NAMESPACE = dict()

CURRENT_SCOPE = ""

NODES = dict()


##############################
# Tracing Wrappers
##############################


class TracedFunctions():
    functions = set()

    @classmethod
    def register_function(cls, function, namespace):
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


class ExplicitUntracedFunctions():
    functions = set()

    @classmethod
    def register_function(cls, function, namespace):
        assert hasattr(namespace, function.__name__)

        traced_function = ExplicitUntracedFunction(namespace, function)

        cls.functions.add(traced_function)

    @classmethod
    def enable(cls):
        for f in cls.functions:
            f.replace_binding()

    @classmethod
    def disable(cls):
        for f in cls.functions:
            f.restore_binding()


class ExplicitUntracedFunction():
    """
    a Wrapper of an arbitrary static function
    which will not be recorded.
    it will not record it's inputs or outputs
    """

    def __init__(self, namespace, original_function):
        self.namespace = namespace
        self.original_function = original_function
        self.function_name = self.original_function.__name__

    def replace_binding(self):
        setattr(self.namespace, self.function_name, self)

    def restore_binding(self):
        setattr(self.namespace,
                self.function_name,
                self.original_function)

    def __call__(self, *args, **kwargs):
        args, kwargs = ExplicitUntracedFunction.ensure_untraced((args, kwargs))

        return self.original_function(*args, **kwargs)

    @staticmethod
    def ensure_untraced(vs):
        def untraced(v):
            if isinstance(v, TracedValue):
                return v._data
            return v

        return nested_map(untraced, vs)


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


def used_namespaces():
    return {namespace.__name__ for namespace in
            chain(get_overridable_functions().keys(), TracedFunctions.traced_namespaces())}


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
        NODES[self.id] = self.node

    def set_data(self, data):
        assert isTracedValue(
            data), f"TracedValue expects a basic type got {type(data)} scope {self.scope}"
        self._ensure_no_hardcoded_device(data)
        self._data = data
        self.namespace = f"{type(self._data).__name__}"
        self.node.value_type = type(data)
        self.node.tensor_dtype = get_tensor_dtypes(data)
        self.node.tensor_shape = get_tensor_shapes(data)

    def _ensure_no_hardcoded_device(self,data):
        if self.node.type is NodeTypes.CONSTANT:
            if isinstance(data,torch.device) or (data == "cpu") or (isinstance(data,str) and "cuda" in data):
                warnings.warn(f"device {data} is hardcoded and cannot be generalized")

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
    #intentionaly explicit
    #NOTE if the method requires specific syntax
    #then it should be also added in utils.py
    # and ensure correct code generation in compiler/partition_forward_method.generate_magic
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

    @delegate_to_traced_value
    def __len__(self):
        pass

        
    ##############################
    # Conversions
    ##############################
    
    #support for conditionals if statements while loops etc.
    #NOTE this must return unwraped value
    # it is prohibited to not return a converted value
    def __bool__(self):
        return bool(self._data)
    

    ##############################
    # Unary operations
    ##############################
    
    @delegate_to_traced_value
    def __neg__(self):
        pass

    @delegate_to_traced_value
    def __pos__(self):
        pass

    @delegate_to_traced_value
    def __abs__(self):
        pass

    @delegate_to_traced_value
    def __invert__(self):
        pass

    ##############################
    # Arithmetic operations
    ##############################

    @delegate_to_traced_value
    def __add__(self, other):
        pass

    @delegate_to_traced_value
    def __sub__(self, other):
        pass

    @delegate_to_traced_value
    def __mul__(self, other):
        pass

    @delegate_to_traced_value
    def __matmul__(self, other):
        pass

    @delegate_to_traced_value
    def __truediv__(self, other):
        pass

    @delegate_to_traced_value
    def __floordiv__(self, other):
        pass

    @delegate_to_traced_value
    def __mod__(self, other):
        pass

    @delegate_to_traced_value
    def __pow__(self, other):
        pass

    @delegate_to_traced_value
    def __lshift__(self, other):
        pass

    @delegate_to_traced_value
    def __rshift__(self, other):
        pass

    @delegate_to_traced_value
    def __and__(self, other):
        pass

    @delegate_to_traced_value
    def __xor__(self, other):
        pass

    @delegate_to_traced_value
    def __or__(self, other):
        pass


    ##############################
    # Reflected Arithmetic operators
    ##############################

    @delegate_to_traced_value
    def __radd__(self, other):
        pass

    @delegate_to_traced_value
    def __rsub__(self, other):
        pass

    @delegate_to_traced_value
    def __rmul__(self, other):
        pass

    @delegate_to_traced_value
    def __rmatmul__(self, other):
        pass

    @delegate_to_traced_value
    def __rtruediv__(self, other):
        pass

    @delegate_to_traced_value
    def __rfloordiv__(self, other):
        pass

    @delegate_to_traced_value
    def __rmod__(self, other):
        pass

    @delegate_to_traced_value
    def __rpow__(self, other):
        pass

    @delegate_to_traced_value
    def __rlshift__(self, other):
        pass

    @delegate_to_traced_value
    def __rrshift__(self, other):
        pass

    @delegate_to_traced_value
    def __rand__(self, other):
        pass

    @delegate_to_traced_value
    def __rxor__(self, other):
        pass

    @delegate_to_traced_value
    def __ror__(self, other):
        pass

    
    ##############################
    # Augmented  Assingment operators
    ##############################

    @delegate_to_traced_value
    def __iadd__(self, other):
        pass

    @delegate_to_traced_value
    def __isub__(self, other):
        pass

    @delegate_to_traced_value
    def __imul__(self, other):
        pass

    @delegate_to_traced_value
    def __imatmul__(self, other):
        pass

    @delegate_to_traced_value
    def __itruediv__(self, other):
        pass

    @delegate_to_traced_value
    def __ifloordiv__(self, other):
        pass

    @delegate_to_traced_value
    def __imod__(self, other):
        pass

    @delegate_to_traced_value
    def __ipow__(self, other):
        pass

    @delegate_to_traced_value
    def __ilshift__(self, other):
        pass

    @delegate_to_traced_value
    def __irshift__(self, other):
        pass

    @delegate_to_traced_value
    def __iand__(self, other):
        pass

    @delegate_to_traced_value
    def __ixor__(self, other):
        pass

    @delegate_to_traced_value
    def __ior__(self, other):
        pass


    ##############################
    # Logical operations
    ##############################
    @delegate_to_traced_value
    def __eq__(self, other):
        pass

    @delegate_to_traced_value
    def __ne__(self, other):
        pass

    @delegate_to_traced_value
    def __ge__(self, other):
        pass

    @delegate_to_traced_value
    def __gt__(self, other):
        pass

    @delegate_to_traced_value
    def __le__(self, other):
        pass

    @delegate_to_traced_value
    def __lt__(self, other):
        pass


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
        self._name = name
        self._module = module
        self._terminal = terminal

    def forward(self, *args, **kwargs):
        global CURRENT_SCOPE
        if CURRENT_SCOPE == "":
            CURRENT_SCOPE = self._name
        else:
            CURRENT_SCOPE += f"/{self._name}"

        args, kwargs = record_args_and_kwargs(*args, **kwargs)

        if self._terminal:
            # NOTE no need to set the creating operation
            # for terminal layer the layer itself is the creating operation
            out = TracedValue(NodeTypes.LAYER, "")

            disable_function_tracing()

            connect_inputs_to_output(out.id, args, kwargs)
            args, kwargs = unpack_traced_args_and_kwargs(*args, **kwargs)

            out.set_data(self._module(*args, **kwargs))

            trace_registered_functions()

        else:
            with record_free_floating_parameters_and_buffers(self._module):
                out = self._module(*args, **kwargs)
                if not isinstance(out, TracedValue):
                    out = record_non_terminal_output(out)

        CURRENT_SCOPE = CURRENT_SCOPE.rsplit("/", maxsplit=1)[0]

        assert isinstance(
            out, TracedValue), f"expected layer output of type TracedValue got {type(out)}"
        return out

    def __getattr__(self, name):
        #NOTE this is different than what we did in TracedValue as layers store buffers/parameters/modules in separate dicts
        try:
            return super().__getattr__(name)
        except Exception:
            return getattr(self._module, name)

    # def __iter__(self):
    #     return iter(self._module)

    def __getitem__(self, key):
        return self._module[key]

    def __setitem__(self, key, value):
        self._module[key] = value

    def __delitem__(self, idx):
        delattr(self._module, idx)

    def __len__(self):
        return len(self._module)

    def __contains__(self, key):
        return key in self._module

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
def trace_module(module: nn.Module, args=(), kwargs=None, depth=1000, basic_blocks=()):
    reset_tracing_state()
    #just to be sure
    _unwrap_layers(module)
    args, kwargs = prepare_args_and_kwargs(args=args, kwargs=kwargs)

    _wrap_traced_layers(module, depth=depth,
                                      basic_blocks=basic_blocks)

    trace_registered_functions()
    ExplicitUntracedFunctions.enable()
    traced_module = TracedLayer(module,
                                name=f"{type(module).__name__}",
                                terminal=False)

    # explicit no grad as we only need the control flow
    with torch.no_grad():
        output = traced_module(*args, **kwargs)
    disable_function_tracing()
    ExplicitUntracedFunctions.disable()

    output_id = output.id

    _unwrap_layers(module)

    for m in module.modules():
        assert not isinstance(m, TracedLayer)

    global CURRENT_SCOPE
    assert CURRENT_SCOPE == traced_module._name

    CURRENT_SCOPE = ""

    nodes = discard_unused_nodes(NODES,output_id)

    # record input kwargs explicitly as they are not passed by position
    # we only retain kwargs that are actually used
    # for example a boolean input that was only used in if checks can be discarded
    input_kw_ids = {v.id: k for k, v in kwargs.items() if v.id in nodes}

    nodes, output_id = set_node_indices(nodes, output_id)
    NODES.clear()

    nodes, output_ids = prepare_graph_outputs(nodes, output_id)

    is_valid, errors = check_is_valid_graph(nodes)
    if not is_valid:
        raise RuntimeError(errors)

    return Graph(nodes, input_kw_ids, output_ids, depth, basic_blocks)


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

    n_args=len(args)
    wrapped_kwargs = dict()
    for i, (k, a) in enumerate(kwargs.items()):
        v = TracedValue(NodeTypes.IN, f"input{n_args+i}")
        v.set_data(a)
        wrapped_kwargs[k] = v

    return wrapped_args, wrapped_kwargs


def register_new_traced_function(function, namespace):
    TracedFunctions.register_function(function, namespace)


def register_new_explicit_untraced_function(function, namespace):
    ExplicitUntracedFunctions.register_function(function, namespace)


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

        if isinstance(sub_layer,(nn.ModuleList,nn.ModuleDict)):
            raise TypeError(f"tracing nn.ModuleList/nn.ModuleDict is not supported got {scope} of type {type(sub_layer)}")
        
        if isinstance(sub_layer,(nn.ParameterList,nn.ParameterDict)):
            # it does not have a forward method so there is nothing to trace
            # we register the parameters for tracing in record_free_floating_parameters_and_buffers
            continue

        wrapper = TracedLayer(sub_layer,
                              scope.rsplit('/', maxsplit=1)[1],
                              terminal)
        parent.add_module(name, wrapper)
        layers_dict[scope] = wrapper

    return layers_dict


def _unwrap_layers(module: nn.Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, TracedLayer):
            _unwrap_layers(sub_module._module)
            module.add_module(name, sub_module._module)
        else:
            module.add_module(name, sub_module)


def reset_tracing_state():
    global CURRENT_SCOPE
    CURRENT_SCOPE = ""
    disable_function_tracing()
    ExplicitUntracedFunctions.disable()
    NODES.clear()
    FUNCTION_NAMESPACE.clear()
    TracedValue.ID = 0


def discard_unused_nodes(nodes,output_id):
    new_nodes = []

    for node_id in reversed(range(len(nodes))):
        node = nodes[node_id]

        if node_id == output_id:
            new_nodes.append((node.id, node))
        
        # if a >1:      a>1 will be traced but it has no meaning to us
        # as we only record the branch that was taken
        unused_branch = False
        if node.type is NodeTypes.OP and (len(node.out_edges)== 0):
            op_path = node.scope.rsplit("/", maxsplit=1)[1]
            _, func_name = op_path.split("::")
            unused_branch = func_name in logical_ops

        # a,b=f() will actually invoke __getitem__ 3 times so we discard the last node
        iter_sentinel = node.value_type is None

        unused_constant_or_input = (node.type in [NodeTypes.IN,NodeTypes.CONSTANT]) and (len(node.out_edges) == 0)

        if unused_branch or iter_sentinel or unused_constant_or_input: 
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
        return nodes, output_id

    new_nodes = dict()

    for idx, node in enumerate(nodes.values()):
        assert idx <= node.id

        node.id = idx
        new_nodes[idx] = node

    return new_nodes, nodes[output_id].id


def prepare_graph_outputs(nodes, output_id):
    """by our convention a graph multiple outputs are represented by multiple nodes
     and not by a single list/tuple node
    """
    output_node = nodes[output_id]

    is_layer_or_tuple = output_node.value_type in {list, tuple}
    is_primitive = output_node.type is NodeTypes.PRIMITIVE

    if not (is_layer_or_tuple and is_primitive):
        return nodes, [output_id]

    # remove the node from the graph
    # set it's parents as outputs
    nodes.pop(output_id)
    output_ids = []
    for n in output_node.in_edges:
        n.remove_output(output_node)
        output_ids.append(n.id)

    return nodes, output_ids

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
            traced_value = TracedValue(NodeTypes.PRIMITIVE,
                                       "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(traced_children))

            for id in traced_ids:
                record_arg(traced_value.id, id)

        elif isinstance(a, dict):
            traced_children, traced_ids = record_kwargs(a,
                                                        top_level=False)
            traced_value = TracedValue(NodeTypes.PRIMITIVE,
                                       "/" + container_construct_op_name(type(a)))
            traced_value.set_data(type(a)(traced_children))

            for k, id in traced_ids.items():
                record_kwarg(traced_value.id, k, id)
        elif isinstance(a, slice):
            traced_children, traced_ids = record_args((a.start, a.stop, a.step),
                                                      top_level=False)
            traced_value = TracedValue(NodeTypes.PRIMITIVE,
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
            traced_value = TracedValue(NodeTypes.PRIMITIVE,
                                       "/" + container_construct_op_name(type(v)))
            traced_value.set_data(type(v)(traced_children))

            for id in children_ids:
                record_arg(traced_value.id, id)

        elif isinstance(v, dict):
            traced_children, traced_ids = record_kwargs(v,
                                                        top_level=False)
            traced_value = TracedValue(NodeTypes.PRIMITIVE,
                                       "/" + container_construct_op_name(type(v)))
            traced_value.set_data(type(v)(traced_children))

            for key, id in traced_ids.items():
                record_kwarg(traced_value.id, key, id)

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
    
    #parameterList/Dict need a special case to ensure correct scope registration
    # as they are modules but do not have a forward method
    for name,c in module.named_children():
        if isinstance(c,(nn.ParameterList,nn.ParameterDict)):
            for p_name,p in c.named_parameters():
                traced_p = TracedValue(NodeTypes.BUFF_PARAM,
                               f"/{type(c).__name__}[{name}]/{type(p).__name__}[{p_name}]")
                
                traced_p.set_data(p)
                c._parameters[p_name] = traced_p
    yield

    # NOTE TracedValue is currently unhashable so we cannot used named_parameters/buffers here
    # so we traverse and modify the state directly
    for name, wrapped_t in chain(module._parameters.items(), module._buffers.items()):
        t = wrapped_t._data
        if isinstance(t, nn.Parameter):
            module._parameters[name] = t
        else:
            module._buffers[name] = t

    #revert parameterList/Dict tracing
    for name,c in module.named_children():
        if isinstance(c,(nn.ParameterList,nn.ParameterDict)):
            for p_name,p in c._parameters.items():
                c._parameters[p_name] = p._data


def record_non_terminal_output(out):
    #NOTE it is possible that a module returns unrecorded outputs
    # like containers None etc. so we ensure that they are all recorded

    recorded_outs,_ = record_args((out,),top_level=True)
    return recorded_outs[0]


def record_kwarg(node_id, kwarg, kwarg_id):
    assert kwarg_id < node_id
    # record the edge

    NODES[kwarg_id].add_out_edge(NODES[node_id])
    NODES[node_id].add_kwarg(kwarg, NODES[kwarg_id])


def record_arg(node_id, arg_id):
    assert arg_id < node_id
    # record the edge

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
                           f"positional args: {[n.id for n in node.args]}",
                           f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
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
                               f"positional args: {[n.id for n in node.args]}",
                               f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

            if o.id < i:
                errors.extend(["violation of topological sort",
                               f"node id: {i}",
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {[n.id for n in node.args]}",
                               f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
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
                               f"positional args: {[n.id for n in node.args]}",
                               f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

            if i < in_node.id:
                errors.extend(["violation of topological sort",
                               f"node id: {i}",
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {[n.id for n in node.args]}",
                               f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
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
                           f"positional args: {[n.id for n in node.args]}",
                           f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
                           f"outgoing edges: {[n.id for n in node.out_edges]}",
                           ""])
            valid = False

        if isinstance(node.tensor_shape, torch.Size) or isinstance(node.tensor_dtype, torch.dtype) or issubclass(node.value_type, Tensor):
            if not ((isinstance(node.tensor_shape, torch.Size)) and (isinstance(node.tensor_dtype, torch.dtype)) and (issubclass(node.value_type, Tensor))):
                errors.extend(["tensor value value not recorded in all of TENSOR_SHAPES TENSOR_DTYPES VALUE_TYPES",
                               f"node id: {i}",
                               f"node id: {i}",
                               f"scope: {node.scope}",
                               f"incoming edges: {[n.id for n in node.in_edges]}",
                               f"positional args: {[n.id for n in node.args]}",
                               f"keyword args: {[(n.id,k) for n,k in node.kwargs.items()]}",
                               f"outgoing edges: {[n.id for n in node.out_edges]}",
                               ""])
                valid = False

    return valid, "\n".join(errors)
