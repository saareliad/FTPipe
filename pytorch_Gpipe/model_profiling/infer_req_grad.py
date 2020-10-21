from typing import Callable, Dict, Any

import torch

from .control_flow_graph import Node, Graph
from .graph_executor import execute_graph
from ..utils import nested_map, detach_tensors


def infer_req_grad(graph: Graph, model: torch.nn.Module, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()

    with torch.enable_grad():
        visitor = Visitor()
        execute_graph(model, graph, model_args=args, model_kwargs=kwargs, pre_hook=visitor.prehook,
                      post_hook=visitor.posthook)


class Visitor():
    def prehook(self, node: Node, function: Callable, args: tuple, kwargs: Dict):
        for n, a in zip(node.args, args):
            # the or statement should not be necessary
            n.req_grad = n.req_grad or Visitor.req_grad(a)

        for n, kws in node.kwargs.items():
            v = kwargs[kws[0]]
            # the or statement should not be necessary
            n.req_grad = n.req_grad or Visitor.req_grad(v)

        return detach_tensors(args), detach_tensors(kwargs)

    def posthook(self, node: Node, function: Callable, args: tuple, kwargs: Dict, outputs: Any):
        node.req_grad = Visitor.req_grad(outputs)

        return detach_tensors(outputs)

    @staticmethod
    def req_grad(ts):
        def f(t):
            if isinstance(t, torch.Tensor):
                return t.requires_grad
            return False

        return nested_map(f, ts)
