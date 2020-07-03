import abc
import operator
from functools import wraps
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import nullcontext
from torch import nn
from pytorch_Gpipe.model_profiling import (Graph, Node, NodeTypes,
                                           used_namespaces)
from pytorch_Gpipe.utils import layerDict, tensorDict,force_out_of_place,inplace_arithmetic_ops


class PreHook(abc.ABC):
    """
    pre hook will be called before the node executes and should have the following signature

    def hook (node: Node, function: Callable, args: tuple, kwargs: dict) -> Tuple[Optional[Tuple], Optional[Dict]]:

    the hook can modify the args/kwargs or return None
    """
    @abc.abstractmethod
    def __call__(self, node: Node, function: Callable, args: tuple, kwargs: dict) -> Tuple[Optional[Tuple], Optional[Dict]]:
        pass


class PostHook(abc.ABC):
    """
    posthook will be called after the node executes and should have the following signature

    def hook (node: Node, function: Callable, args: tuple, kwargs: dict,outputs) ->Optional:

    the hook can modify the output or return None
    """
    @abc.abstractmethod
    def __call__(self, node: Node, function: Callable, args: tuple, kwargs: Dict, outputs: Any) -> Optional:
        pass


def execute_graph(model: nn.Module, graph: Graph, model_args=(), model_kwargs=None, pre_hook: Optional[PreHook] = None, post_hook: Optional[PostHook] = None,enforce_out_of_place=True):
    if model_kwargs is None:
        model_kwargs = dict()
    if not isinstance(model_args, tuple):
        model_args = (model_args,)

    if pre_hook is None:
        pre_hook = IdentityPreHook()

    if post_hook is None:
        post_hook = IdentityPostHook()

    pre_hook = apply_pre_hook(pre_hook)
    post_hook = apply_post_hook(post_hook)

    nodes: List[Node] = sorted(graph.nodes, key=lambda n: n.id)

    # node uses
    uses = {n: len(n.out_edges) for n in nodes}
    for n in graph.outputs:
        uses[n] += 1

    # prepare inputs including kwargs
    ready_expressions = dict(zip(nodes, model_args))
    for node in graph.inputs:
        if node.id in graph.input_kw_ids:
            ready_expressions[node] = model_kwargs[graph.input_kw_ids[node.id]]

    del model_args
    del model_kwargs

    # prepare buffs params
    tensors = tensorDict(model)
    ready_expressions.update({n: tensors[n.scope] for n in nodes
                              if n.type is NodeTypes.BUFF_PARAM})
    del tensors

    layers = layerDict(model, graph.depth, graph.basic_blocks)
    namespaces = used_namespaces()
    for node in nodes:
        if node in ready_expressions:
            continue

        if node.type is NodeTypes.CONSTANT:
            v = node.constant_value

            ready_expressions[node] = v
            continue

        args, kwargs = fetch_args_kwargs(node, ready_expressions)

        if node.type is NodeTypes.LAYER:
            l = layers[node.scope]
            with force_out_of_place(l) if enforce_out_of_place else nullcontext():
                args, kwargs = pre_hook(node, l, args, kwargs)
                outputs = l(*args, **kwargs)
                outputs = post_hook(node, l, args, kwargs, outputs)

            ready_expressions[node] = outputs

        elif node.type is NodeTypes.PRIMITIVE:

            ready_expressions[node] = create_container_construct(node,
                                                                 args,
                                                                 kwargs)
        else:
            assert node.type is NodeTypes.OP
            outputs = call_function(namespaces,
                                    node,
                                    args, kwargs,
                                    pre_hook, post_hook,enforce_out_of_place=enforce_out_of_place)

            ready_expressions[node] = outputs
        del args
        del kwargs
        # ensure we discard intermediate values as soon as possible to conserve memory
        for n in node.in_edges:
            uses[n] -= 1
            if uses[n] == 0:
                ready_expressions.pop(n)

        # in case this value is not being used anywhere
        # for example used,_ = f()
        # the second output should be profiled but we should not hold on to it
        if uses[node] == 0:
            ready_expressions.pop(node)

    return [ready_expressions[n] for n in graph.outputs]


def create_container_construct(node, args, kwargs):
    if "prim::DictConstruct" in node.scope:
        return kwargs
    elif "prim::SetConstruct" in node.scope:
        return set(args)
    elif "prim::ListConstruct" in node.scope:
        return list(args)
    elif "prim::TupleConstruct" in node.scope:
        return tuple(args)
    else:
        assert "prim::SliceConstruct" in node.scope
        return slice(*args)


def call_function(namespaces, node, args, kwargs, pre_hook, post_hook,enforce_out_of_place=True):
    op_path = node.scope.rsplit("/", maxsplit=1)[1].rsplit("_",maxsplit=1)[0]
    namespace, func_name = op_path.split("::")
    # function call
    if namespace in namespaces:
        namespace = import_module(namespace)
        function = getattr(namespace, func_name)
    else:
        if "__" not in func_name:
            function = getattr(type(args[0]), func_name)
        elif func_name == "__getattribute__":
            # __getattribute__ is implemented for all python objects
            return getattr(args[0], args[1])
        else:
            assert len(kwargs) == 0, "no kwarg in magic method"

            if enforce_out_of_place and (func_name in inplace_arithmetic_ops):
                func_name ="__"+func_name[3:]
            if hasattr(operator, func_name):
                function = getattr(operator, func_name)
            else:
                function = getattr(type(args[0]), func_name)

    args, kwargs = pre_hook(node, function, args, kwargs)
    output = function(*args, **kwargs)
    output = post_hook(node, function, args, kwargs, output)

    return output


def fetch_args_kwargs(node, ready_expressions):
    args = [ready_expressions[n] for n in node.args]
    kwargs = {k: ready_expressions[n] for n, k in node.kwargs.items()}

    return args, kwargs


def apply_pre_hook(pre_hook):
    @wraps(pre_hook)
    def wrapper(node: Node, function: Callable, args: tuple, kwargs: dict):
        modified_args, modified_kwargs = pre_hook(node, function, args, kwargs)
        if not (modified_args is None):
            args = modified_args
        if not (modified_kwargs is None):
            kwargs = modified_kwargs

        return args, kwargs

    return wrapper


def apply_post_hook(post_hook):
    @wraps(post_hook)
    def wrapper(node: Node, function: Callable, args: tuple, kwargs: dict, outputs):
        modified_outputs = post_hook(node, function, args, kwargs, outputs)
        if not (modified_outputs is None):
            outputs = modified_outputs
        return outputs

    return wrapper


class IdentityPreHook(PreHook):
    def __call__(self, node: Node, function: Callable, args: tuple, kwargs: dict) -> Tuple[Optional[Tuple], Optional[Dict]]:
        return args, kwargs


class IdentityPostHook(PostHook):
    def __call__(self, node: Node, function: Callable, args: tuple, kwargs: Dict, outputs: Any) -> Optional:
        return outputs
