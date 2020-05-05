import operator
from typing import List
from importlib import import_module
from torch import nn
from pytorch_Gpipe.model_profiling import Node, NodeTypes, Graph, used_namespaces
from pytorch_Gpipe.utils import tensorDict, layerDict


def execute_graph(model: nn.Module, graph: Graph, args=(), kwargs=None):
    if kwargs is None:
        kwargs = dict()
    if not isinstance(args, tuple):
        args = tuple(args)

    nodes: List[Node] = sorted(graph.nodes, key=lambda n: n.id)

    # node uses
    uses = {n: len(n.out_edges) for n in nodes}
    for n in graph.outputs:
        uses[n] += 1

    # prepare inputs including kwargs
    ready_expressions = dict(zip(nodes, args))
    for node in graph.inputs:
        if node.id in graph.input_kw_ids:
            ready_expressions[node] = kwargs[graph.input_kw_ids[node.id]]

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
            ready_expressions[node] = node.constant_value
            continue

        if node.type is NodeTypes.LAYER:
            print(f"layer call {node.scope}")
            l = layers[node.scope]
            args, kwargs = fetch_args_kwargs(node, ready_expressions)
            ready_expressions[node] = l(*args, **kwargs)

        elif node.type is NodeTypes.PRIMITIVE:
            print(f"building container {node.value_type} {node.scope}")
            args, kwargs = fetch_args_kwargs(node, ready_expressions)
            ready_expressions[node] = create_container_construct(node,
                                                                 args,
                                                                 kwargs)
        else:
            assert node.type is NodeTypes.OP
            ready_expressions[node] = call_function(ready_expressions,
                                                    namespaces,
                                                    node)

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


def call_function(ready_expressions, namespaces, node):
    op_path = node.scope.rsplit("/", maxsplit=1)[1]
    namespace, func_name = op_path.split("::")

    # function call
    if namespace in namespaces:
        args, kwargs = fetch_args_kwargs(node, ready_expressions)

        print(f"calling {op_path}")
        namespace = import_module(namespace)
        function = getattr(namespace, func_name)
        output = function(*args, **kwargs)

    else:
        args, kwargs = fetch_args_kwargs(node, ready_expressions)
        self_arg = args[0]

        if "__" not in func_name:
            print(f"calling instance method {op_path}")
            instance_method = getattr(self_arg, func_name)
            output = instance_method(*(args[1:]),
                                     **kwargs)

        elif func_name == "__getattribute__":
            # __getattribute__ is implemented for all python objects
            output = getattr(self_arg, args[1])
        else:
            assert len(kwargs) == 0, "no kwarg in magic method"
            if hasattr(operator, func_name):
                print(f"calling magic {op_path} using operator")
                magic_method = getattr(operator, func_name)
                output = magic_method(*args)
            else:
                print(f"calling magic {op_path} using the instance")
                magic_method = getattr(self_arg, func_name)
                output = magic_method(*(args[1:]))
    return output


def fetch_args_kwargs(node, ready_expressions):
    args = [ready_expressions[n] for n in node.args]
    kwargs = {k: ready_expressions[n] for n, k in node.kwargs.items()}

    return args, kwargs
