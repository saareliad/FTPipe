from typing import List
import torch
from torch import Tensor
from torch.nn import functional
from .control_flow_graph import Graph, Node, NodeTypes
from .network_profiler import Profile
from ..compiler.partition_forward_method import SupportedFunctions, getAtenFunctionNameAndScope


NAMESPACES = {"torch": torch,
              "Tensor": Tensor,
              "F": functional}


def profile_ops(graph: Graph):
    for node in graph.nodes:
        if node.type is NodeTypes.OP:
            node.weight = profile_op(get_op(node.scope),
                                     create_inputs(node.in_nodes))


def get_op(scope: str):
    function, namespace = getAtenFunctionNameAndScope(scope)
    return getattr(NAMESPACES[namespace], function)


def create_inputs(in_nodes: List[Node]):
    inputs = []

    for n in in_nodes:
        if n.type is NodeTypes.CONSTANT:
            pass
        else:
            assert n.shape and n.dtype


def profile_op(op, inputs) -> Profile:
    print(op)
    assert torch.cuda.is_available(), "Profiling ops supports only GPU profiling"
    return Profile
