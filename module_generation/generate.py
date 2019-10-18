
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes, Graph
import string
from .forward import generateForwardFunction, PartitionIO
from .constructor import generateConstructor
from pprint import pprint
from typing import List, Tuple, Dict
from pytorch_Gpipe.utils import OrderedSet
from collections import OrderedDict
import inspect


def generatePartitionModules(graph: Graph, layer_classes: Dict[str, Module], is_param_dict: Dict[str, bool], verbose=False) -> Tuple[List[str], List[PartitionIO]]:
    parts = groupByPartition(graph.nodes)

    lines = generatePytorchImports(layer_classes)

    ios = []

    # the main code generation loop generating a class decl
    # and forward function
    for idx, part in parts:
        class_name = f'{graph.model_name}Partition{idx}'
        layer_names = [n.scope for n in part if n.type == NodeTypes.LAYER]
        buff_param_names = {n.scope for n in part
                            if n.type == NodeTypes.BUFF_PARAM}
        class_decl, scope_to_class_field = generateConstructor(class_name, layer_names,
                                                               layer_classes, is_param_dict,
                                                               buff_param_names)
        forward_function, io = generateForwardFunction(part,
                                                       scope_to_class_field, verbose=verbose)
        lines.append(class_decl)
        lines.extend(forward_function)
        ios.append(io)

    return lines, ios


def groupByPartition(nodes: List[Node]) -> List[Tuple[int, List[Node]]]:
    # groups layers and their respective nodes according to their partition
    idxs = {n.part for n in nodes}
    parts = OrderedDict()
    for i in sorted(idxs):
        parts[i] = []

    for n in nodes:
        if n.type == NodeTypes.IN:
            continue
        elif n.type == NodeTypes.BUFF_PARAM:
            parts[n.part].append(n)
        elif n.type == NodeTypes.LAYER:
            parts[n.part].append(n)
        elif n.type == NodeTypes.OP:
            scope = n.scope
            # we handle torch,Tensor and torch.nn.functional nameSpaces
            func_name = getFunctionName(scope)
            if hasattr(torch, func_name) or hasattr(F, func_name) or hasattr(Tensor, func_name):
                parts[n.part].append(n)
            else:
                assert False, f'could not find nameSpace for {scope}'
        elif n.type == NodeTypes.PYTHON_PRIMITIVE:
            scope = n.scope
            assert 'prim::' in scope, f'primitive does not have prim:: prefix {scope}'
            assert 'ListConstruct' in scope, f'expected ListConstruct got {scope}'
            func_name = scope.split('prim::')[1].rstrip(string.digits)
            parts[n.part].append(n)
        else:
            assert n.type == NodeTypes.CONSTANT, f'got type {n.type}'
            parts[n.part].append(n)

    return parts.items()


def generatePytorchImports(layer_classes: Dict[str, Module]) -> List[str]:
    '''generates imports to torch torch.nn, torch.nn.functionl as F and torch.Tensor,
       and to every layer used
    '''
    imports = f'import torch\nfrom torch import Tensor\nimport torch.nn as nn\nimport torch.nn.functional as F\n'

    unique_classes = set(layer_classes.values())

    for cls in unique_classes:
        imports += f'from {inspect.getmodule(cls).__name__} import {cls.__name__}\n'

    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'

    return imports.splitlines() + [disclaimer]


def getFunctionName(scope: str) -> str:
    if 'aten::' in scope:
        sep = 'aten::'
    else:
        assert 'prim::' in scope, f"attempting to find function name but got {scope}"
        sep = 'prim::'

    return scope.split(sep)[1].rstrip(string.digits)
