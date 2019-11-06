
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes, Graph
from pytorch_Gpipe.utils import traverse_model, traverse_params_buffs
import string
from .forward import generateForwardFunction
from .constructor import generateConstructor
from .misc import generateMiscMethods
from typing import List, Tuple, Dict
from pytorch_Gpipe.utils import OrderedSet
from collections import OrderedDict
import inspect

tab = '    '
dtab = tab + tab


def generatePartitionModules(graph: Graph, model: Module, verbose=False, output_file=None, DEBUG=False):
    layer_classes = {scope: type(layer) for layer, scope, _
                     in traverse_model(model, depth=graph.depth)}
    is_param_dict = {scope: t.requires_grad for t,
                     scope in traverse_params_buffs(model)}

    parts = groupByPartition(graph.nodes)

    lines = generateImports(layer_classes)
    lines.append(connections(graph))
    ios = []
    # the main code generation loop generating a class decl
    # and forward function
    partitions_code = []
    ios = dict()
    for idx, part in parts:
        class_name = f'{graph.model_name}Partition{idx}'
        layer_names = [n.scope for n in part if n.type == NodeTypes.LAYER]
        buff_param_names = {n.scope for n in part
                            if n.type == NodeTypes.BUFF_PARAM}
        class_decl, scope_to_class_field = generateConstructor(class_name, layer_names,
                                                               layer_classes, is_param_dict,
                                                               buff_param_names)
        misc_functions = generateMiscMethods()
        forward_function, io = generateForwardFunction(part, graph.output_scopes, scope_to_class_field,
                                                       verbose=verbose)
        partitions_code.append(class_decl)
        partitions_code.extend(forward_function)
        partitions_code.append(misc_functions)
        ios[idx] = io

    lines.append(generatePipline(graph, parts, model, ios, DEBUG=DEBUG))
    lines += partitions_code

    if output_file is None:
        output_file = f'generated_{graph.model_name}{len(parts)}'

    output_file = output_file + '.py'

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


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
            elif 'aten::slice' in scope:
                parts[n.part].append(n)
            else:
                assert False, f'could not find nameSpace for {scope}'
        elif n.type == NodeTypes.PYTHON_PRIMITIVE:
            scope = n.scope
            assert 'prim::' in scope, f'primitive does not have prim:: prefix {scope}'
            func_name = scope.split('prim::')[1].rstrip(string.digits)
            parts[n.part].append(n)
        else:
            assert n.type == NodeTypes.CONSTANT, f'got type {n.type}'
            parts[n.part].append(n)

    return parts.items()


def generateImports(layer_classes: Dict[str, Module]) -> List[str]:
    '''generates imports to torch torch.nn, torch.nn.functionl as F and torch.Tensor,
       and to every layer used and various other small things
    '''
    imports = 'import torch\nfrom torch import Tensor\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
    imports += 'from itertools import chain\n'
    imports += 'from pytorch_Gpipe.utils import layerDict, tensorDict, OrderedSet\n'
    imports += 'from pytorch_Gpipe import Pipeline\n'
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


def generatePipline(graph: Graph, partitions: List[List[Node]], model: Module, ios: Dict[int, OrderedSet], DEBUG=False):
    '''generates function that will perform the actual partition returning a Pipeline object\n
       the function will have the partition config hardcoded into it,\n
       enabling us to perform the partition process once and use the config multiple times
    '''
    model_buffers = {scope: t for t, scope in traverse_params_buffs(model)
                     if not t.requires_grad}
    model_parameteres = {scope: t for t, scope in traverse_params_buffs(model)
                         if t.requires_grad}
    model_class = model.__class__.__name__
    # function header
    lines = [
        f'def {model_class}Pipeline(model:nn.Module,output_device=None,use_delayedNorm=False,DEBUG=False):',
        "layer_dict = layerDict(model)",
        "tensor_dict = tensorDict(model)",
        f"\n{tab}# now constructing the partitions in order"
    ]
    lines = [f'\n{tab}'.join(lines)]

    # hard code which layers buffers and parameters belong to each partition
    construction_args = []
    for idx, part in partitions:
        layer_scopes = [f"'{n.scope}'"
                        for n in part if n.type == NodeTypes.LAYER]
        buffer_scopes = [
            f"'{n.scope}'" for n in part if n.scope in model_buffers]
        parameter_scopes = [f"'{n.scope}'" for n in part
                            if n.scope in model_parameteres]
        construction_args.append(
            (layer_scopes, buffer_scopes, parameter_scopes))

    # create partition generation statements
    for idx, (layer_scopes, buffer_scopes, parameter_scopes) in zip(sorted(list(ios.keys())), construction_args):
        l_scopes = 'layer_scopes = [' + f",\n{dtab}".join(layer_scopes) + ']'
        b_scopes = 'buffer_scopes = [' + f",\n{dtab}".join(buffer_scopes) + ']'
        p_scopes = 'parameter_scopes = [' + \
            f",\n{dtab}".join(parameter_scopes) + ']'
        lines.extend([l_scopes, b_scopes, p_scopes,
                      f"layers = {{l: layer_dict[l] for l in layer_scopes}}",
                      f"buffers = {{b: tensor_dict[b] for b in buffer_scopes}}",
                      f"parameters = {{p: tensor_dict[p] for p in parameter_scopes}}",
                      f"partition{idx} = {model_class}Partition{idx}(layers,buffers,parameters)\n"])

    # create and return the partition config
    exp = f',\n{dtab}{tab}'.join([f"{k}: {v}" for k, v in ios.items()])
    lines.append(
        f"# creating configuration\n{tab}config = {{{exp}\n{dtab}{tab}}}")

    for idx in sorted(list(ios.keys())):
        device = f'cuda:{idx}' if not DEBUG else 'cpu'
        lines.append(
            f"config[{idx}]['model'] = partition{idx}.to('{device}')")

    input_ids = [f"'input{idx}'" for idx in range(graph.num_inputs)]
    lines.extend([f"config['model inputs'] = [{', '.join(input_ids)}]",
                  f"config['model outputs'] = {list(graph.output_scopes)}"])

    lines.append(
        f"\n{tab}return Pipeline(config,output_device=output_device,use_delayedNorm=use_delayedNorm,DEBUG=DEBUG)\n\n")

    return f'\n{tab}'.join(lines)


def connections(graph: Graph):
    adj_matrix = [{"inputs": set(), "outputs": set()}
                  for i in range(graph.num_parts + 2)]

    for node in graph.nodes:
        if node.idx < graph.num_inputs:
            for n in node.out_nodes:
                adj_matrix[n.part + 1]["inputs"].add(node.scope)
                adj_matrix[0]["outputs"].add(n.part)

        idx = graph.output_scopes.indexOf(node.scope)

        if idx >= 0:
            adj_matrix[graph.num_parts + 1]["inputs"].add(node.part)
            adj_matrix[node.part + 1]["outputs"].add(f"output{idx}")

        for n in node.out_nodes:
            if n.part != node.part:
                adj_matrix[node.part + 1]["outputs"].add(n.part)
                adj_matrix[n.part + 1]["inputs"].add(node.part)

    lines = ["# partition adjacency"]
    lines.append(f"# model inputs {adj_matrix[0]['outputs']}")
    for i, line in enumerate(adj_matrix[1:-1:]):
        lines.append(f"# partition {i} {line}")
    lines.append(
        f"# model outputs {adj_matrix[graph.num_parts + 1]['inputs']}")
    return '\n'.join(lines) + '\n'
