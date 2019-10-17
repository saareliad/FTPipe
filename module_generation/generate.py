
import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes, Graph
import string
from .forward import generateForwardFunction, PartitionIO
from .constructor import generateConstructor
from pprint import pprint
from typing import List, Tuple


def generatePartitionModules(graph: Graph) -> Tuple[List[str], List[PartitionIO]]:
    parts = groupByPartition(graph.nodes)

    torch_import = generatePytorchImports()

    lines = [torch_import]
    ios = []

    # the main code generation loop generating a class decl
    # and forward function
    for idx, part in enumerate(parts):
        # TODO there are empty partitions which is obviously not good
        class_name = f'{graph.model_name}Partition{idx}'
        names = [n.scope for n in part if n.type == NodeTypes.LAYER]
        class_decl, scope_to_class_field = generateConstructor(class_name,
                                                               names)
        forward_function, io = generateForwardFunction(part,
                                                       scope_to_class_field)
        lines.extend([class_decl, forward_function])
        ios.append(io)

    return lines, ios


def groupByPartition(nodes: List[Node]) -> List[List[Node]]:
    # groups layers and their respective nodes according to their partition
    # TODO if we have less partitions not all indices will appear
    parts = [[] for i in range(len({n.part for n in nodes}))]
    for n in nodes:
        if n.type == NodeTypes.IN or n.type == NodeTypes.BUFF_PARAM:
            # TODO handle in buff param
            continue
        if n.type == NodeTypes.LAYER:
            try:
                parts[n.part].append(n)
            except Exception as _:
                print(
                    f'when adding layer {scope} invalid partition idx {n.part}')
        elif n.type == NodeTypes.OP:
            scope = n.scope
            # we handle torch,Tensor and torch.nn.functional nameSpaces
            func_name = scope.split('aten::')[1].rstrip(string.digits)
            if hasattr(torch, func_name) or hasattr(F, func_name) or hasattr(Tensor, func_name):
                try:
                    parts[n.part].append(n)
                except Exception as _:
                    print(
                        f'when adding op{scope} invalid partition idx {n.part}')
            else:
                assert False, f'could not find nameSpace for {scope}'
        elif n.type == NodeTypes.PYTHON_PRIMITIVE:
            scope = n.scope
            assert 'prim::' in scope, f'primitive does not have prim:: prefix {scope}'
            func_name = scope.split('prim::')[1].rstrip(string.digits)
            assert func_name == 'ListConstruct'
            try:
                parts[n.part].append(n)
            except Exception as _:
                print(
                    f'when adding prim{scope} invalid partition idx {n.part}')
        else:
            assert n.type == NodeTypes.CONSTANT
            try:
                parts[n.part].append(n)
            except Exception as _:
                print(
                    f'when adding Constant{scope} invalid partition idx {n.part}')

    return parts


def generatePytorchImports() -> str:
    '''generates imports to torch torch.nn, torch.nn.functionl as F and torch.Tensor
    '''
    imports = f'import torch\nfrom torch import Tensor\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'

    return imports + disclaimer
