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
from .utils import format_shape
tab = '    '
dtab = tab + tab


def compile_partitoned_model(graph: Graph,
                             model: Module,
                             batch_dim: int,
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
        create_pipeline_configuration(graph, parts, model, ios, layer_classes, batch_dim, output_file))
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
                                  model_blocks: Dict[str, Module], batch_dim: int, output_file: str) -> str:
    '''generates the create_pipeline_configuration method which given a model creates his partitioned counterpart
    '''
    module_path = output_file.replace("/", ".")
    model_buffers = {
        scope: t
        for t, scope in traverse_params_buffs(model) if not t.requires_grad
    }
    model_parameteres = {
        scope: t
        for t, scope in traverse_params_buffs(model) if t.requires_grad
    }
    model_class = model.__class__.__name__
    basic_blocks = ",".join(
        map(lambda block: block.__name__, set(model_blocks.values())))

    serialized_basic_blocks = f",\n{dtab}{tab}".join(f"'{inspect.getmodule(cls).__name__}.{cls.__name__}'"
                                                     for cls in set(model_blocks.values()))

    # function header
    lines = [
        f"def create_pipeline_configuration(DEBUG=False):",
        f"depth = {graph.depth}",
        f"basic_blocks = ({basic_blocks})",
        f"blocks_path = [ {serialized_basic_blocks}]",
        f"module_path = '{module_path}'",
        "\n"
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
        lines.extend([l_scopes, b_scopes, p_scopes, "\n"])

    # create and return the partition config
    def format_dict(d):
        items = [f'"{k}":{v}' for k, v in d.items()]
        return "{" + f",\n{dtab}".join(items) + "}"

    exp = f',\n{dtab}{tab}'.join(
        [f"'{k}': {format_dict(v)}" for k, v in ios.items()])
    lines.append(
        f"# creating configuration\n{tab}stages = {{{exp}\n{dtab}{tab}}}")

    for idx in sorted(list(ios.keys())):
        lines.extend(["\n",
                      f"stages['{idx}']['batch_dim'] = {batch_dim}",
                      f"stages['{idx}']['stage_cls'] = module_path + '.Partition{idx}'",
                      f"device = 'cpu' if DEBUG else'cuda:{idx}'",
                      f"stages['{idx}']['devices'] = [device]",
                      f"stages['{idx}']['optimizer'] = {{'type':'','args':{{}}}}",
                      f"stages['{idx}']['lr_scheduler'] = {{'type':'','args':{{}}}}",
                      ])

    input_ids = [f"'input{idx}'" for idx in range(graph.num_inputs)]
    input_shapes = [format_shape(n.shape)[0] for n in graph.inputs]
    model_outputs = graph.outputs
    output_shapes = [format_shape(n.shape)[0] for n in model_outputs]

    lines.extend([
        "\n",
        "config = dict()",
        f"config['batch_dim'] = {batch_dim}",
        f"config['depth'] = depth",
        f"config['basic_blocks'] = blocks_path",
        f"config['model_inputs'] = [{', '.join(input_ids)}]",
        f"config['model_input_shapes'] = {input_shapes}",
        f"config['model_outputs'] = {list(graph.output_scopes)}",
        f"config['model_output_shapes'] = {output_shapes}",
        f"config['stages'] = stages",
        f"\n{tab}return config"
    ])
    return f"\n{tab}".join(lines) + "\n"


def connections(graph: Graph) -> str:
    '''creates a diagram that illustrates the connectins between partitions,
    to be embeded in the generated file
    '''
    num_partitions = graph.num_partitions
    adj_matrix = [{"inputs": set(), "outputs": set()}
                  for i in range(num_partitions + 2)]

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
    lines.append(
        f"# model outputs {adj_matrix[num_partitions + 1]['inputs']}")
    return '\n'.join(lines) + '\n'


def create_model_parallel_module(name: str, ios: Dict[int, Dict[str,
                                                                List[str]]],
                                 num_inputs: int,
                                 model_outputs: List[str]) -> str:
    '''create a modelParallel version of the partition config
    '''
    class_decl_and_init = "\n".join([
        f"class ModelParallel(nn.Module):",
        f"{tab}def __init__(self,layers,tensors,CPU=False):",
        f"{dtab}super(ModelParallel,self).__init__()",
        dtab + f"\n{dtab}".join(f"self.stage{i} = Partition{i}(layers,tensors).to('cpu' if CPU else 'cuda:{i}')"
                                for i in ios)
    ])

    model_inputs = [f'input{idx}' for idx in range(num_inputs)]
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

    cnt = 0
    while len(parts) > 0:
        if cnt > 3 * n_partitions:
            assert False, "error cycle detected mutual dependecy between generated partitions"
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
            cnt += 1
            parts.append(idx)

    outputs = ", ".join(activations[o] for o in model_outputs)
    statements.append(f"return {outputs}")

    return f"\n{dtab}".join(statements)
