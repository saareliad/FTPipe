from typing import List, Callable, Tuple, Any
import torch
from torch import Tensor
from torch.nn import functional
from .control_flow_graph import Graph, Node, NodeTypes
from .network_profiler import Profile
from ..compiler.partition_forward_method import SupportedFunctions, getAtenFunctionNameAndScope


NAMESPACES = {"torch": torch,
              "Tensor": Tensor,
              "F": functional}


class Closure():
    def __call__(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


class ConstantClosure(Closure):
    def __init__(self, v):
        super(ConstantClosure, self).__init__()
        self.v = v

    def __call__(self):
        return self.v

    def __str__(self):
        return str(self.v)


class TensorClosure(Closure):
    def __init__(self, shape, dtype, device='cuda'):
        super(TensorClosure, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)

    def __call__(self):
        return torch.empty(size=self.shape, dtype=self.dtype, device=self.device)

    def __str__(self):
        return f"Tensor shape {self.shape} dtype: {self.dtype} device {self.device}"


class ListClosure(Closure):
    def __init__(self, args: List[Closure]):
        super(ListClosure, self).__init__()
        self.closures = args

    def __call__(self):
        return [c() for c in self.closures]

    def __getitem__(self, idx):
        return self.closures[idx]

    def __str__(self):
        return "[" + ", ".join(str(c) for c in self.closures) + "]"


def profile_ops(graph: Graph):
    cache = dict()
    for node in graph.nodes:
        if node.type is NodeTypes.OP:
            node.weight, value = profile_op(get_op(node.scope),
                                            [cache[i] for i in node.in_nodes])
        elif node.type is NodeTypes.CONSTANT:
            if isinstance(node.value, torch.device):
                value = torch.device("cuda")
            else:
                value = ConstantClosure(node.value)
        elif node.valueType() is Tensor:
            value = TensorClosure(node.shape[0], node.dtype[0])
        elif "prim::TupleConstruct" in node.scope or "prim::ListConstruct" in node.scope:
            value = ListClosure([cache[i] for i in node.in_nodes])
        elif node.type is NodeTypes.PYTHON_PRIMITIVE:
            assert "Unpack" in node.scope
            father = node.in_nodes[0]
            idx = father.out_nodes.indexOf(node)
            value = cache[father][idx]
        else:
            # list/tuple layer or input
            assert node.type in [NodeTypes.LAYER, NodeTypes.IN]
            assert node.valueType() in [list, tuple]
            value = createNestedTensorList(node.shape, node.dtype)

        cache[node] = value

    for k, v in cache.items():
        print(k.scope, str(v))


def createNestedTensorList(shapes, dtypes):
    if isinstance(shapes, torch.Size):
        return TensorClosure(shapes, dtypes)

    return ListClosure([createNestedTensorList(s, d) for s, d in zip(shapes, dtypes)])


def get_op(scope: str) -> Callable:
    function, namespace = getAtenFunctionNameAndScope(scope)
    return getattr(NAMESPACES[namespace], function)


def profile_op(op, inputs) -> Tuple[Profile, Any]:
    # print(op, inputs)
    assert torch.cuda.is_available(), "Profiling ops supports only GPU profiling"

    return Profile, ConstantClosure("Not supported")
