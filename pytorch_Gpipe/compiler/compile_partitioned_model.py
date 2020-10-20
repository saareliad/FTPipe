import inspect
import os
import pathlib
import warnings
from collections import OrderedDict
from typing import List, Dict, Optional, Iterable
import networkx as nx
from functools import reduce

from torch.nn import Module

from pytorch_Gpipe.utils import traverse_model, traverse_params_buffs, layerDict, tensorDict, nested_map, move_tensors, \
    flatten, _unflatten, unflatten
# from .partition_class import Partition
from .compile_modelParallel_module import create_model_parallel_module
from .create_pipeline_configuration import create_pipeline_configuration
from .partition_forward_method import generate_forward_method
from .partition_init_method import generate_init_method
from .state_methods import get_state_methods, generate_partition_state_methods
from .utils import ensure_no_unnecessary_tuple_sends, ensure_inputs_are_used
from ..model_profiling import Node, NodeTypes, Graph, used_namespaces

tab = '    '
dtab = tab + tab


def compile_partitioned_model(graph: Graph,
                              model: Module,
                              batch_dim: int,
                              generate_model_parallel: bool = False,
                              generate_explicit_del=False,
                              generate_activation_propagation=True,
                              output_file: Optional[str] = None,
                              move_tensors=False):
    """
    generates the code for the partitioned model.
       The partitions can be consumed using the `create_pipeline_configuration` method in the generated code

    Parameters:
    graph:
        the partitioned graph of the module
    module:
        the module itself
    batch_dim:
        the batch dimension of the input
    generate_model_parallel:
        whether to generate a model parallel version of the partition in the addition to the partitions themselves
    generate_explicit_del:
        whether to generate del statements to explicitly delete variables when they are no longer used
        default False
    generate_activation_propagation:
        in cases where a stage sends an activation to multiple stages.
        for example 0->[1,3,4]
        decide weather to have each stage send the activation to the next target
        0->1->3->4
        or have it sent directly from the source
    output_file:
        optional path to the generated code. if None uses generated_{model_name}{numberOfPatitions}.py
    """
    n1 = graph.num_partitions
    ensure_inputs_are_used(graph)
    n2 = graph.num_partitions
    ensure_no_unnecessary_tuple_sends(graph)
    n3 = graph.num_partitions
    # Assert that we don't accidentally kill stages.
    assert n1 == n2, f"'ensure_inputs_are_used' accidentally killed a stage {(n1,n2)}"
    assert n2 == n3, f"'ensure_no_unnecessary_tuple_sends' accidentally killed a stage {(n2,n3)}"

    layer_classes = {
        scope: type(layer)
        for layer, scope, _ in traverse_model(
            model, depth=graph.depth, basic_blocks=graph.basic_blocks)
    }
    is_param_dict = {
        scope: t.requires_grad
        for t, scope in traverse_params_buffs(model)
    }

    stages = group_nodes_by_stage_id(graph.nodes)

    stage_depth_from_end = get_stages_depth_from_end(graph)
    # for stage_id, stage_nodes in stages.items():
    #     for n in stage_nodes:
    #         n.depth = stage_depth_from_end[stage_id]

    lines = generate_imports(layer_classes)
    lines.append(stage_connections_str(graph))
    # the main code generation loop generating a class decl
    # and forward function
    partitions_code = []
    ios = dict()
    for idx, stage_nodes in stages.items():
        class_name = f'Partition{idx}'
        layers = [n for n in stage_nodes if n.type == NodeTypes.LAYER]
        buffs_params = [
            n
            for n in stage_nodes if n.type == NodeTypes.BUFF_PARAM
        ]
        class_decl, scope_to_class_field = generate_init_method(stage_nodes, class_name, layers,
                                                                is_param_dict, buffs_params)
        state_methods_functions = generate_partition_state_methods()
        forward_function, io = generate_forward_method(idx,
                                                       graph,
                                                       stage_nodes,
                                                       graph.outputs,
                                                       scope_to_class_field,
                                                       stage_depth_from_end=stage_depth_from_end[idx],
                                                       generate_explicit_del=generate_explicit_del,
                                                       generate_activation_propagation=generate_activation_propagation,
                                                       move_tensors=False)
        partitions_code.append(class_decl)
        partitions_code.extend(forward_function)
        partitions_code.append("")
        partitions_code.append(state_methods_functions)
        ios[idx] = io

    if output_file is None:
        output_file = f'generated_{graph.model_name}{len(stages)}'
    elif output_file.endswith(".py"):
        output_file = output_file[:-3]

    create_pipeline_configuration_str, config = create_pipeline_configuration(graph, ios, layer_classes, batch_dim,
                                                                              generate_activation_propagation)
    lines.append(create_pipeline_configuration_str)
    if generate_model_parallel:
        lines.append(create_model_parallel_module(config))

    # lines .append(inspect.getsource(Partition))
    lines += partitions_code
    lines.append(generate_help_functions())

    path = pathlib.Path(output_file + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        os.remove(path)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def group_nodes_by_stage_id(nodes: Iterable[Node]) -> Dict[int, List[Node]]:
    """
    Groups nodes by their stage_id
    """
    idxs = {n.stage_id for n in nodes}
    stages = OrderedDict()
    for i in sorted(idxs):
        stages[i] = []

    for n in nodes:
        stages[n.stage_id].append(n)
    return stages


def generate_imports(layer_classes: Dict[str, Module]) -> List[str]:
    """
    generates imports to torch torch.nn, torch.nn.functionl as F and torch.Tensor,
       and to every layer used and various other small things
    """
    imports = [f'import {namespace}' for namespace in used_namespaces()]
    imports.extend(['from torch import Tensor',
                    'import torch.nn as nn',
                    'from itertools import chain',
                    'from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict',
                    'import collections',
                    # 'from abc import ABC, abstractmethod'
                    ''])
    unique_classes = set(layer_classes.values())

    for cls in unique_classes:
        imports.append(
            f'from {inspect.getmodule(cls).__name__} import {cls.__name__}')

    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'
    imports.append(disclaimer)

    return imports


def generate_help_functions() -> str:
    """generates traverse_model, layerDict, traverse_params_buffs, tensorDict functions,
    to be used in the create_pipeline_configuration function and
    parameters,named_parameters,buffers,named_buffers,cpu,cuda,to,state_dict,load_state_dict
    to be used by the partitions themselves
    """
    lines = [
        inspect.getsource(f) for f in
        [traverse_model, layerDict, traverse_params_buffs,
         tensorDict, move_tensors, nested_map, flatten, unflatten, _unflatten] + get_state_methods()
    ]

    return "\n\n".join(lines)


def stage_connections_str(graph: Graph) -> str:
    """creates a diagram that illustrates the connectivity between partitions,
    to be embedded in the generated file
    """
    adj_matrix, num_partitions = stages_adj_lists(graph)

    lines = ["# partition adjacency",
             f"# model inputs {adj_matrix[0]['outputs']}"]
    for i, line in enumerate(adj_matrix[1:-1:]):
        lines.append(f"# partition {i} {line}")
    lines.append(
        f"# model outputs {adj_matrix[num_partitions + 1]['inputs']}")

    n_output_stages = len(adj_matrix[num_partitions + 1]['inputs'])
    if n_output_stages != 1:
        warnings.warn(f"Got {n_output_stages} output stages, expected 1")
    return '\n'.join(lines) + '\n'


def stages_adj_lists(graph):
    num_partitions = graph.num_partitions
    adj_matrix = [{"inputs": set(), "outputs": set()}
                  for i in range(num_partitions + 2)]
    for node in graph.nodes:
        if node.type is NodeTypes.IN:
            for n in node.out_edges:
                adj_matrix[n.stage_id + 1]["inputs"].add(graph.input_kw_ids.get(node.id, node.scope))
                adj_matrix[0]["outputs"].add(n.stage_id)
            continue

        if node in graph.outputs:
            adj_matrix[num_partitions + 1]["inputs"].add(node.stage_id)
            adj_matrix[node.stage_id + 1]["outputs"].add(f"output")

        for n in node.out_edges:
            if n.stage_id != node.stage_id:
                adj_matrix[node.stage_id + 1]["outputs"].add(n.stage_id)
                adj_matrix[n.stage_id + 1]["inputs"].add(node.stage_id)

    return adj_matrix, num_partitions


def dict_stages_adj_lists(graph):
    adj_matrix, num_partitions = stages_adj_lists(graph)
    dict_adj_matrix = {}
    keys = ['model_inputs'] + list(range(num_partitions)) + ['model_outputs']
    for key, v in zip(keys, adj_matrix):
        dict_adj_matrix[key] = v

    return dict_adj_matrix, num_partitions


def get_stages_depth_from_end(graph) -> Dict[int, int]:
    dict_adj_matrix, num_partitions = dict_stages_adj_lists(graph)

    # reverse graph
    edges = set()
    for i, d in dict_adj_matrix.items():
        if i in {"model_inputs", "model_outputs"}:
            continue
        for x in d['inputs']:
            edges.add((i, x))
        for x in d['outputs']:
            edges.add((x, i))

    G = nx.DiGraph(list(edges))

    # Can do it much more efficiently with dynamic programing, but its a tiny graph so its meaningless
    def longest_depth_length(target):
        return reduce(max, map(len, nx.all_simple_edge_paths(G, source=f"output", target=target))) - 1

    distance_dict = {i: longest_depth_length(i) for i in range(num_partitions)}

    for i,v in distance_dict.items():
        if v < 0:
            warnings.warn(f"Stage {i} was not used in output calculation. distance_dict={distance_dict}")

    if len(set(distance_dict.values())) < num_partitions:
        warnings.warn(f"Detected parallel stages. Naive pipelines can't run this. distance_dict={distance_dict}")

    return distance_dict
