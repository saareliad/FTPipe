from typing import Callable, Dict, Any

import torch

from .control_flow_graph import Node, Graph
from .graph_executor import execute_graph, pre_hook_factory, post_hook_factory
from ..utils import nested_map, detach_tensors


def infer_is_contiguous(graph: Graph, model: torch.nn.Module, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()

    with torch.no_grad():
        visitor = Visitor()
        execute_graph(model, graph, model_args=args, model_kwargs=kwargs, pre_hook=pre_hook_factory(visitor.prehook),
                      post_hook=post_hook_factory(visitor.posthook))


class Visitor():
    def prehook(self, node: Node, function: Callable, args: tuple, kwargs: Dict):
        for n, a in zip(node.args, args):
            # the or statement should not be necessary
            n.is_contiguous = n.is_contiguous or Visitor.is_contiguous(a)

        for n, kws in node.kwargs.items():
            v = kwargs[kws[0]]
            # the or statement should not be necessary
            n.is_contiguous = n.is_contiguous or Visitor.is_contiguous(v)

        return detach_tensors(args), detach_tensors(kwargs)

    def posthook(self, node: Node, function: Callable, args: tuple, kwargs: Dict, outputs: Any):
        node.is_contiguous = Visitor.is_contiguous(outputs)

        return detach_tensors(outputs)

    @staticmethod
    def is_contiguous(ts):
        def f(t):
            if isinstance(t, torch.Tensor):
                return t.is_contiguous()
            return False

        return nested_map(f, ts)
