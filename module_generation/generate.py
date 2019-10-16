
import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch_Gpipe.model_profiling import NodeTypes
from pytorch_Gpipe import partition_with_profiler
import string
from .forward import generate_forward_function
from .constructor import generate_constructor
from pprint import pprint


def generate_modules(model: torch.nn.Module, *sample_batch, depth=0):
    # TODO the graph should be a parameter
    graph = partition_with_profiler(model, *sample_batch,
                                    max_depth=depth, nparts=4)

    model_name = type(model).__name__
    graph.save(model_name, '.', show_buffs_params=True, show_weights=True)
    parts = group_by_partition(graph.nodes)

    # import torch torch.nn as nn import torch.nn.functional as F
    torch_import = generate_imports()

    lines = [torch_import]

    # the main code generation loop generating a class decl
    # and forward function
    for idx, part in enumerate(parts):
        # TODO there are empty partitions which is obviously not good
        class_name = f'{model_name}Partition{idx}'
        names = [n.scope for n in part if n.type == NodeTypes.LAYER]
        class_decl, scope_to_id = generate_constructor(class_name, names)
        forward, partitionIO = generate_forward_function(part, scope_to_id)

        print("inputs")
        pprint(partitionIO.inputs)
        print()

        print("outputs")
        pprint(partitionIO.outputs)
        print()
        lines.extend([class_decl, forward])

    return lines


def group_by_partition(nodes):
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


def generate_imports():
    imports = f'import torch\nfrom torch import Tensor\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'

    return imports + disclaimer
