import torch
from torch.nn import Module
from ..model_profiling import Node, NodeTypes, Graph, used_namespaces
from pytorch_Gpipe.utils import traverse_model, traverse_params_buffs, layerDict, tensorDict,nested_map,move_tensors
from .partition_forward_method import generate_forward_method
from .partition_init_method import generate_init_method
from .state_methods import get_state_methods, generate_partition_state_methods
from .compile_modelParallel_module import create_model_parallel_module
from typing import List, Tuple, Dict, Optional, Callable
from collections import OrderedDict
import inspect
import os
import pathlib
tab = '    '
dtab = tab + tab


def compile_partitioned_model(graph: Graph,
                              model: Module,
                              batch_dim: int,
                              generate_model_parallel: bool = False,
                              generate_explicit_del=False,
                              output_file: Optional[str] = None):
    '''generates the code for the partitioned model.
       The partitions can be consumed using the `create_pipeline_configuration` method in the generated code

    Parameters:
    graph:
        the partitoned graph of the module
    module:
        the module itself
    batch_dim:
        the batch dimention of the input
    generate_model_parallel:
        whether to generate a model parallel version of the partition in the addition to the partitions themselves
    generate_explicit_del:
        whether to generate del statements to explicitly delete variables when they are no longer used
        default False
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
        layers = [n for n in part if n.type == NodeTypes.LAYER]
        buffs_params = [
            n
            for n in part if n.type == NodeTypes.BUFF_PARAM
        ]
        class_decl, scope_to_class_field = generate_init_method(class_name, layers,
                                                                is_param_dict, buffs_params)
        state_methods_functions = generate_partition_state_methods()
        forward_function, io = generate_forward_method(graph,
                                                       part,
                                                       graph.outputs,
                                                       scope_to_class_field,
                                                       generate_explicit_del=generate_explicit_del)
        partitions_code.append(class_decl)
        partitions_code.extend(forward_function)
        partitions_code.append("")
        partitions_code.append(state_methods_functions)
        ios[idx] = io

    if output_file is None:
        output_file = f'generated_{graph.model_name}{len(parts)}'
    elif output_file.endswith(".py"):
        output_file = output_file[:-3]

    lines.append(
        create_pipeline_configuration(graph, ios, layer_classes, batch_dim))
    if generate_model_parallel:
        lines.append(
            create_model_parallel_module(graph, batch_dim, ios, graph.num_inputs,
                                         graph.output_scopes))
    lines += partitions_code
    lines.append(generateHelpFunctions())

    path = pathlib.Path(output_file + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        os.remove(path)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def groupByPartition(nodes: List[Node]) -> List[Tuple[int, List[Node]]]:
    '''groups nodes to their respective partitions
    '''
    idxs = {n.part for n in nodes}
    parts = OrderedDict()
    for i in sorted(idxs):
        parts[i] = []

    for n in nodes:
        parts[n.part].append(n)
    return parts.items()


def generateImports(layer_classes: Dict[str, Module]) -> List[str]:
    '''generates imports to torch torch.nn, torch.nn.functionl as F and torch.Tensor,
       and to every layer used and various other small things
    '''
    imports = [f'import {namespace}' for namespace in used_namespaces()]
    imports.extend(['from torch import Tensor',
                    'import torch.nn as nn',
                    'from itertools import chain',
                    'from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict',
                    'import collections',
                    'import os'
                    ''])
    unique_classes = set(layer_classes.values())

    for cls in unique_classes:
        imports.append(
            f'from {inspect.getmodule(cls).__name__} import {cls.__name__}')

    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'
    imports.append(disclaimer)

    return imports


def generateHelpFunctions() -> str:
    '''generates traverse_model, layerDict, traverse_params_buffs, tensorDict functions,
    to be used in the create_pipeline_configuration function and
    parameters,named_parameters,buffers,named_buffers,cpu,cuda,to,state_dict,load_state_dict
    to be used by the partitions themselves
    '''
    lines = [
        inspect.getsource(f) for f in
        [traverse_model, layerDict, traverse_params_buffs,
            tensorDict,move_tensors,nested_map] + get_state_methods()
    ]

    return "\n\n".join(lines)


def create_pipeline_configuration(graph: Graph,
                                  ios: Dict[int,
                                            Dict[str,
                                                 List[str]]],
                                  model_blocks: Dict[str, Module],
                                  batch_dim: int) -> str:
    '''generates the create_pipeline_configuration method which given a model creates his partitioned counterpart
    '''
    # TODO assumption the first input is batched
    batch_size = graph._nodes[0].tensor_shape[batch_dim]

    # TODO better logic for is_batched
    # currently we assume that if a tensor has enough dimentions and the relevant dim size equals the batch_size
    def is_batched(s):
        return (s is not None) and(len(s) > (batch_dim + 1)) and (s[batch_dim] == batch_size)

    module_path = 'os.path.relpath(__file__).replace("/",".")[:-3]'
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
        f"module_path = {module_path}",
        "\n"
    ]

    # create and return the partition config

    def format_dict(d):
        items = [f'"{k}": {v}' for k, v in d.items()]
        return "{" + f",\n{dtab}".join(items) + "}"

    exp = f',\n{dtab}{tab}'.join(
        [f"{k}: {format_dict(v)}" for k, v in stages_in_out_config(ios, is_batched).items()])
    lines.append(
        f"# creating configuration\n{tab}stages = {{{exp}\n{dtab}{tab}}}")

    for idx in sorted(list(ios.keys())):
        lines.extend(["\n",
                      f"stages[{idx}]['stage_cls'] = module_path + '.Partition{idx}'",
                      f"device = 'cpu' if DEBUG else 'cuda:{idx}'",
                      f"stages[{idx}]['devices'] = [device]",
                      ])

    model_inputs, model_outputs = create_model_in_out_config(graph, is_batched)

    model_inputs = f',\n{dtab}{tab}'.join([f"{k}: {format_dict(v)}"
                                           for k, v in model_inputs.items()])
    model_outputs = f',\n{dtab}{tab}'.join([f"'{k}': {format_dict(v)}"
                                            for k, v in model_outputs.items()])

    lines.extend([
        "\n",
        "config = dict()",
        f"config['batch_dim'] = {batch_dim}",
        f"config['depth'] = depth",
        f"config['basic_blocks'] = blocks_path",
        f"config['model_inputs'] = {{{model_inputs}}}",
        f"config['model_outputs'] = {{{model_outputs}}}",
        f"config['stages'] = stages",
        f"\n{tab}return config"
    ])
    return "\n" + f"\n{tab}".join(lines) + "\n"


def connections(graph: Graph) -> str:
    '''creates a diagram that illustrates the connectins between partitions,
    to be embeded in the generated file
    '''
    num_partitions = graph.num_partitions
    adj_matrix = [{"inputs": set(), "outputs": set()}
                  for i in range(num_partitions + 2)]

    for node in graph.nodes:
        if node.type is NodeTypes.IN:
            for n in node.out_edges:
                adj_matrix[n.part + 1]["inputs"].add(node.scope)
                adj_matrix[0]["outputs"].add(n.part)

        if node in graph.outputs:
            adj_matrix[num_partitions + 1]["inputs"].add(node.part)
            adj_matrix[node.part + 1]["outputs"].add(f"output")

        for n in node.out_edges:
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


def stages_in_out_config(ios: Dict, is_batched: Callable[[torch.Size], bool]) -> Dict:
    '''generates the stages portion of the config
     stages:
       id
            stage_inputs
                id
                shape
                dtype
                is_batched
                req_grad
            stage_outputs
                id
                shape
                dtype
                is_batched
    '''
    config = dict()

    for idx, io in ios.items():
        inputs = io['inputs']
        outputs = io['outputs']
        input_shapes = io['input_shapes']
        input_dtypes = io['input_dtypes']
        output_dtypes = io['output_dtypes']
        output_shapes = io['output_shapes']
        inputs_req_grad = io['inputs_req_grad']

        stage_inputs = dict()
        for i, s, r, d in zip(inputs, input_shapes,inputs_req_grad, input_dtypes):
            stage_inputs[i] = {"shape": s,
                               "dtype": d,
                               "req_grad":r,
                               "is_batched": is_batched(s)}
        # TODO it is possible that if we have a single output
        # it's still a list/tuple for example return l(x) where l returns multiple outputs
        stage_outputs = dict()
        for o, s, d in zip(outputs, output_shapes, output_dtypes):
            stage_outputs[o] = {"shape": s,
                                "dtype": d,
                                "is_batched": is_batched(s)}

        config[idx] = {"inputs": stage_inputs,
                       "outputs": stage_outputs}
    return config


def create_model_in_out_config(graph: Graph, is_batched: Callable[[torch.Size], bool]) -> Tuple[Dict, Dict]:
    """create the config of model inputs and outputs
        model_inputs
            id
                shape
                dtype
                is_batched
        model_outputs
            id
                shape
                dtype
                is_batched
    """

    input_ids = [f"'{graph.input_kw_ids.get(node.id,node.scope)}'" for node in graph.inputs]
    input_shapes = [n.tensor_shape for n in graph.inputs]
    input_dtypes = [n.tensor_dtype for n in graph.inputs]
    # TODO it is possible that if we have a single output
    # it's still a list/tuple for example return l(x) where l returns multiple outputs
    output_shapes = [n.tensor_shape for n in graph.outputs]
    output_ids = graph.output_scopes
    output_dtypes = [n.tensor_dtype for n in graph.outputs]

    model_inputs = dict()
    for i, s, d in zip(input_ids, input_shapes, input_dtypes):
        model_inputs[i] = {"shape": s,
                           "dtype": d,
                           "is_batched": is_batched(s)}

    model_outputs = dict()
    for o, s, d in zip(output_ids, output_shapes, output_dtypes):
        model_outputs[o] = {"shape": s,
                            "dtype": d,
                            "is_batched": is_batched(s)}

    return model_inputs, model_outputs
