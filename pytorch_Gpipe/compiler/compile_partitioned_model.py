import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes, Graph
from pytorch_Gpipe.utils import traverse_model, traverse_params_buffs, layerDict, tensorDict
import string
from .partition_forward_method import generate_forward_method, variableNameGenerator
from .partition_init_method import generate_init_method
from .state_methods import generate_state_methods
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict, deque
import inspect
import os
import pathlib

tab = '    '
dtab = tab + tab


def compile_partitoned_model(graph: Graph,
                             model: Module,
                             verbose: bool = False,
                             output_file: Optional[str] = None):
    '''generates the code for the partitioned model.
       The partitions can be consumed using the `create_pipeline_configuration` method in the generated code

    Parameters:
    graph:
        the partitoned graph of the module
    module:
        the module itself
    verbose:
        whether to generate each statement of the forward function in a seperate line (usefull for debugging)
    output_file:
        optional path to the generated code. if None uses generated_{model_name}{numberOfPatitions}.py
    '''
    layer_classes = {
        scope: type(layer)
        for layer, scope, _ in traverse_model(
            model, depth=graph.depth, basic_blocks=graph.basic_blocks)
    }
    is_param_dict = {
        scope: t.requires_grad
        for t, scope in traverse_params_buffs(model)
    }

    parts = groupByPartition(graph.nodes)

    lines = generateImports(layer_classes)
    lines.append(connections(graph))
    ios = []
    # the main code generation loop generating a class decl
    # and forward function
    partitions_code = []
    ios = dict()
    for idx, part in parts:
        class_name = f'Partition{idx}'
        layer_names = [n.scope for n in part if n.type == NodeTypes.LAYER]
        buff_param_names = {
            n.scope
            for n in part if n.type == NodeTypes.BUFF_PARAM
        }
        class_decl, scope_to_class_field = generate_init_method(
            class_name, layer_names, layer_classes, is_param_dict,
            buff_param_names)
        state_methods_functions = generate_state_methods()
        forward_function, io = generate_forward_method(part,
                                                       graph.output_scopes,
                                                       scope_to_class_field,
                                                       verbose=verbose)
        partitions_code.append(class_decl)
        partitions_code.extend(forward_function)
        partitions_code.append(state_methods_functions)
        ios[idx] = io

    lines.append(
        create_pipeline_configuration(graph, parts, model, ios, layer_classes))
    lines.append(
        create_model_parallel_module(graph.model_name, ios, graph.num_inputs,
                                     graph.output_scopes))
    lines += partitions_code
    lines.append(generateHelpFunctions())

    if output_file is None:
        output_file = f'generated_{graph.model_name}{len(parts)}'

    output_file = output_file + '.py'

    path = pathlib.Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        os.remove(path)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def groupByPartition(nodes: List[Node]) -> List[Tuple[int, List[Node]]]:
    '''groups nodes to their respective partitions for OP Nodes ensure we recognize thier namespace
    '''
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
            if hasattr(torch, func_name) or hasattr(F, func_name) or hasattr(
                    Tensor, func_name):
                parts[n.part].append(n)
            elif 'aten::slice' in scope or 'aten::Int' in scope:
                parts[n.part].append(n)
            else:
                assert False, f'could not find nameSpace for {scope}'
        elif n.type == NodeTypes.PYTHON_PRIMITIVE:
            scope = n.scope
            assert 'prim::' in scope, f'primitive does not have prim:: prefix {scope}'
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
    imports += 'import operator\n'
    imports += 'from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict\n'
    imports += 'import collections'
    imports += '\n'
    unique_classes = set(layer_classes.values())

    for cls in unique_classes:
        imports += f'from {inspect.getmodule(cls).__name__} import {cls.__name__}\n'

    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'

    return imports.splitlines() + [disclaimer]


def generateHelpFunctions() -> str:
    '''generates traverse_model, layerDict, traverse_params_buffs, tensorDict functions
    to be used in the create_pipeline_configuration function
    '''
    lines = [
        inspect.getsource(f) for f in
        [traverse_model, layerDict, traverse_params_buffs, tensorDict]
    ]

    return "\n\n".join(lines)


def getFunctionName(scope: str) -> str:
    '''returns the name of a function belonging to the aten/prim namespaces
    '''
    if 'aten::' in scope:
        sep = 'aten::'
    else:
        assert 'prim::' in scope, f"attempting to find function name but got {scope}"
        sep = 'prim::'

    return scope.split(sep)[1].rstrip(string.digits)


def create_pipeline_configuration(graph: Graph, partitions: List[List[Node]],
                                  model: Module, ios: Dict[int,
                                                           Dict[str,
                                                                List[str]]],
                                  basic_blocks: Dict[str, Module]) -> str:
    '''generates the create_pipeline_configuration method which given a model creates his partitioned counterpart
    '''
    model_buffers = {
        scope: t
        for t, scope in traverse_params_buffs(model) if not t.requires_grad
    }
    model_parameteres = {
        scope: t
        for t, scope in traverse_params_buffs(model) if t.requires_grad
    }
    model_class = model.__class__.__name__
    if graph.basic_blocks:
        basic_blocks = [cls.__name__ for cls in set(basic_blocks.values())]
    else:
        basic_blocks = []
    if len(basic_blocks) == 1:
        basic_blocks = f"{basic_blocks[0]},"
    else:
        basic_blocks = ",".join(basic_blocks)

    # function header
    lines = [
        f"def create_pipeline_configuration(model,DEBUG=False,partitions_only=False):",
        f"layers = layerDict(model,depth={graph.depth},basic_blocks=({basic_blocks}))",
        "tensors = tensorDict(model)",
        f"\n{tab}# now constructing the partitions in order"
    ]

    # hard code which layers buffers and parameters belong to each partition
    construction_args = []
    for idx, part in partitions:
        layer_scopes = [
            f"'{n.scope}'" for n in part if n.type == NodeTypes.LAYER
        ]
        buffer_scopes = [
            f"'{n.scope}'" for n in part if n.scope in model_buffers
        ]
        parameter_scopes = [
            f"'{n.scope}'" for n in part if n.scope in model_parameteres
        ]
        construction_args.append(
            (layer_scopes, buffer_scopes, parameter_scopes))

    # create partition generation statements
    for idx, (layer_scopes, buffer_scopes,
              parameter_scopes) in zip(sorted(list(ios.keys())),
                                       construction_args):
        l_scopes = 'layer_scopes = [' + f",\n{dtab}".join(layer_scopes) + ']'
        b_scopes = 'buffer_scopes = [' + f",\n{dtab}".join(buffer_scopes) + ']'
        p_scopes = 'parameter_scopes = [' + \
            f",\n{dtab}".join(parameter_scopes) + ']'
        lines.extend([l_scopes, b_scopes, p_scopes,
                      f"partition{idx} = Partition{idx}(layers,tensors)\n"])

    # create and return the partition config
    exp = f',\n{dtab}{tab}'.join([f"{k}: {v}" for k, v in ios.items()])
    lines.append(
        f"# creating configuration\n{tab}config = {{{exp}\n{dtab}{tab}}}")

    for idx in sorted(list(ios.keys())):
        lines.extend([
            f"device = torch.device('cpu') if DEBUG else torch.device('cuda:{idx}')",
            f"config[{idx}]['model'] = partition{idx}.to(device)"
        ])

    input_ids = [f"'input{idx}'" for idx in range(graph.num_inputs)]
    lines.extend([
        f"config['model inputs'] = [{', '.join(input_ids)}]",
        f"config['model outputs'] = {list(graph.output_scopes)}",
        f"\n{tab}return [config[i]['model'] for i in range({len(ios)})] if partitions_only else config"
    ])

    return f"\n{tab}".join(lines) + "\n"


def connections(graph: Graph) -> str:
    '''creates a diagram that illustrates the connectins between partitions,
    to be embeded in the generated file
    '''
    num_partitions = graph.num_partitions
    adj_matrix = [{
        "inputs": set(),
        "outputs": set()
    } for i in range(num_partitions + 2)]

    for node in graph.nodes:
        if node.type is NodeTypes.IN:
            for n in node.out_nodes:
                adj_matrix[n.part + 1]["inputs"].add(node.scope)
                adj_matrix[0]["outputs"].add(n.part)

        idx = graph.output_scopes.indexOf(node.scope)

        if idx >= 0:
            adj_matrix[num_partitions + 1]["inputs"].add(node.part)
            adj_matrix[node.part + 1]["outputs"].add(f"output{idx}")

        for n in node.out_nodes:
            if n.part != node.part:
                adj_matrix[node.part + 1]["outputs"].add(n.part)
                adj_matrix[n.part + 1]["inputs"].add(node.part)

    lines = ["# partition adjacency"]
    lines.append(f"# model inputs {adj_matrix[0]['outputs']}")
    for i, line in enumerate(adj_matrix[1:-1:]):
        lines.append(f"# partition {i} {line}")
    lines.append(f"# model outputs {adj_matrix[num_partitions + 1]['inputs']}")
    return '\n'.join(lines) + '\n'


def create_model_parallel_module(name: str, ios: Dict[int, Dict[str,
                                                                List[str]]],
                                 num_inputs: int,
                                 model_outputs: List[str]) -> str:
    '''create a modelParallel version of the partition config
    '''
    model_inputs = [f'input{idx}' for idx in range(num_inputs)]
    class_decl_and_init = "\n".join([
        f"class ModelParallel(nn.Module):",
        f"{tab}def __init__(self,config):",
        f"{dtab}super({name}ModelParallel,self).__init__()",
        dtab + f"\n{dtab}".join(f"self.stage{i} = config[{i}]['model']"
                                for i in ios)
    ])

    forward = model_parallel_forward(ios, model_inputs, model_outputs)

    states = f",\n{dtab}{dtab}".join(
        [f"**self.stage{i}.state_dict(self.stage{i}.device)" for i in ios])

    states = f"{{{states}}}"

    state_dict = f"\n{dtab}".join(
        ["def state_dict(self):", f"return {states}"])

    loads = f"\n{dtab}".join([f"self.stage{i}.load_state(state)" for i in ios])
    load_state_dict = f"\n{tab}{tab}".join(
        ["def load_state_dict(self,state):", loads])

    buffer_states = f",\n{dtab}{dtab}{tab} ".join(
        [f"self.stage{i}.named_buffers()" for i in ios])
    named_buffers = f"\n{dtab}".join(
        [f"def named_buffers(self):", f"return chain({buffer_states})"])

    parameter_states = f",\n{dtab}{dtab}{tab} ".join(
        [f"self.stage{i}.named_parameters()" for i in ios])
    named_parameters = f"\n{dtab}".join(
        [f"def named_parameters(self):", f"return chain({parameter_states})"])

    parameters = f"\n{dtab}".join([
        "def parameters(self):",
        f"return [p for _,p in self.named_parameters()]"
    ])

    buffers = f"\n{dtab}".join(
        ["def buffers(self):", f"return [b for _,b in self.named_buffers()]"])

    return f"\n\n{tab}".join([
        class_decl_and_init, forward, state_dict, load_state_dict,
        named_buffers, named_parameters, buffers, parameters
    ]) + "\n\n"


def model_parallel_forward(ios: Dict[int, Dict[str, List[str]]],
                           model_inputs: List[str],
                           model_outputs: List[str]) -> str:
    '''generates the forward nethod of the model parallel version of the config
    '''
    n_partitions = len(ios)
    arg_gen = variableNameGenerator()
    activations = {}
    statements = [f"def forward(self,{', '.join(model_inputs)}):"]

    for idx, i in enumerate(model_inputs):
        activations[i] = f'input{idx}'

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        if all(tensor in activations for tensor in ios[idx]['inputs']):
            inputs = ", ".join(
                f"{activations[tensor]}.to(self.stage{idx}.device)"
                for tensor in ios[idx]['inputs'])
            outputs = []
            for o, t in zip(ios[idx]['outputs'], arg_gen):
                activations[o] = t
                outputs.append(t)

            outputs = ", ".join(outputs)

            statements.append(f"{outputs} = self.stage{idx}({inputs})")
            if len(ios[idx]['outputs']) == 1:
                statements[-1] += '[0]'

        else:
            parts.append(idx)

    outputs = ", ".join(activations[o] for o in model_outputs)
    statements.append(f"return {outputs}")

    return f"\n{dtab}".join(statements)
