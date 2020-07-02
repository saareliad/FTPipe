from typing import Callable,Dict
import torch
from .graph_executor import execute_graph
from .control_flow_graph import Node,NodeTypes,Graph
from ..utils import nested_map,detach_tensors


def infer_req_grad(graph:Graph,model:torch.nn.Module,args=None,kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()
    
    visitor = Visitor()
    execute_graph(model,graph,model_args=args,model_kwargs=kwargs,pre_hook=visitor,post_hook=None)



class Visitor():
    def __call__(self, node: Node, function: Callable, args: tuple, kwargs: Dict):
        for n,a in zip(node.args,args):
            if n.stage_id != node.stage_id or n.type is NodeTypes.IN:
                n.req_grad = Visitor.req_grad(a)
            
        for n,k in node.kwargs.items():
            v = kwargs[k]
            n.req_grad = Visitor.req_grad(v)

        return detach_tensors(args),detach_tensors(kwargs)


    @staticmethod
    def req_grad(ts):
        def f(t):
            if isinstance(t,torch.Tensor):
                return t.requires_grad
            return False
        
        return nested_map(f,ts)

