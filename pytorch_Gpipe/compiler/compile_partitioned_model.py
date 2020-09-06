import inspect
import os
import pathlib
from collections import OrderedDict
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Callable

import torch
from torch.nn import Module

from pytorch_Gpipe.utils import traverse_model, traverse_params_buffs, layerDict, tensorDict, nested_map, move_tensors, \
    flatten, _unflatten, unflatten
from .compile_modelParallel_module import create_model_parallel_module
from .partition_forward_method import generate_forward_method
from .partition_init_method import generate_init_method
from .state_methods import get_state_methods, generate_partition_state_methods
from .utils import pretty_format_obj, ensure_inputs_are_used
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

    ensure_inputs_are_used(graph)

    layer_classes = {
        scope: type(layer)
        for layer, scope, _ in traverse_model(
            model, depth=graph.depth, basic_blocks=graph.basic_blocks)
    }
    is_param_dict = {
        scope: t.requires_grad
        for t, scope in traverse_params_buffs(model)
    }

    stages = groupByPartition(graph.nodes)

    lines = generateImports(layer_classes)
    lines.append(connections(graph))
    ios = []
    # the main code generation loop generating a class decl
    # and forward function
    partitions_code = []
    ios = dict()
    for idx, stage in stages:
        class_name = f'Partition{idx}'
        layers = [n for n in stage if n.type == NodeTypes.LAYER]
        buffs_params = [
            n
            for n in stage if n.type == NodeTypes.BUFF_PARAM
        ]
        class_decl, scope_to_class_field = generate_init_method(stage, class_name, layers,
                                                                is_param_dict, buffs_params)
        state_methods_functions = generate_partition_state_methods()
        forward_function, io = generate_forward_method(idx,
                                                       graph,
                                                       stage,
                                                       graph.outputs,
                                                       scope_to_class_field,
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
    idxs = {n.stage_id for n in nodes}
    stages = OrderedDict()
    for i in sorted(idxs):
        stages[i] = []

    for n in nodes:
        stages[n.stage_id].append(n)
    return stages.items()


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
         tensorDict, move_tensors, nested_map, flatten, unflatten, _unflatten] + get_state_methods()
    ]

    return "\n\n".join(lines)


def create_pipeline_configuration(graph: Graph,
                                  ios: Dict[int,
                                            Dict[str,
                                                 List[str]]],
                                  model_blocks: Dict[str, Module],
                                  batch_dim: int,
                                  generate_activation_propagation: bool) -> Tuple[str, Dict]:
    '''generates the create_pipeline_configuration method which given a model creates his partitioned counterpart
    '''
    # TODO assumption the first input is batched
    batch_size = sorted(graph.inputs, key=lambda n: n.id)[0].tensor_shape[batch_dim]

    def is_batched(shape):
        def f(s):
            return (s is not None) and (len(s) > (batch_dim + 1)) and (s[batch_dim] == batch_size)

        return nested_map(f, shape)

    basic_blocks = ",".join(
        map(lambda block: block.__name__, set(model_blocks.values())))

    # function header
    lines = [
        f"def create_pipeline_configuration(DEBUG=False, batch_size={batch_size}):"
    ]

    # create and return the partition config
    model_inputs, model_outputs = create_model_in_out_config(graph, is_batched)
    stages = create_stages_config(ios, is_batched)
    config = {
        "batch_dim": batch_dim,
        "depth": graph.depth,
        "basic_blocks": f'({basic_blocks})',
        "model_inputs": model_inputs,
        "model_outputs": model_outputs,
        "stages": stages
    }

    if generate_activation_propagation:
        # modify the config to accomodate input propagation
        config = generate_config_with_input_propagation(config)

    config = generate_config_without_nested(config)

    lines.extend([
        f"config = {pretty_format_obj(config)}",
        ""
    ])

    lines.append(f"\n{tab}# switching batch size")
    lines.append(generate_switch_batch_size())

    lines.append(f"\n{tab}return config")
    return "\n" + f"\n{tab}".join(lines) + "\n", config


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

    lines = ["# partition adjacency"]
    lines.append(f"# model inputs {adj_matrix[0]['outputs']}")
    for i, line in enumerate(adj_matrix[1:-1:]):
        lines.append(f"# partition {i} {line}")
    lines.append(
        f"# model outputs {adj_matrix[num_partitions + 1]['inputs']}")
    return '\n'.join(lines) + '\n'


def create_stages_config(ios: Dict, is_batched: Callable[[torch.Size], bool]) -> Dict:
    '''generates the stages portion of the config
     stages:
       id
            stage_cls
            stage_inputs
                id
                shape
                dtype
                is_batched
                req_grad
                created_by
            stage_outputs
                id
                shape
                dtype
                is_batched
                req_grad
                used_by
            devices
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
        outputs_req_grad = io['outputs_req_grad']
        created_by = io['created_by']
        used_by = io['used_by']

        stage_inputs = dict()
        for i, s, r, d, src in zip(inputs, input_shapes, inputs_req_grad, input_dtypes, created_by):
            stage_inputs[i] = {"shape": s,
                               "dtype": d,
                               "req_grad": r,
                               "is_batched": is_batched(s),
                               "created_by": src}

        stage_outputs = dict()
        for o, s, r, d, dsts in zip(outputs, output_shapes, outputs_req_grad, output_dtypes, used_by):
            stage_outputs[o] = {"shape": s,
                                "dtype": d,
                                "req_grad": r,
                                "is_batched": is_batched(s),
                                "used_by": dsts}

        config[idx] = {"stage_cls": f"Partition{idx}",
                       "inputs": stage_inputs,
                       "outputs": stage_outputs,
                       "devices": f"['cpu' if DEBUG else 'cuda:{idx}']"}
    return config


def create_model_in_out_config(graph: Graph, is_batched: Callable[[torch.Size], bool]) -> Tuple[Dict, Dict]:
    """create the config of model inputs and outputs
        model_inputs
            id
                shape
                dtype
                is_batched
                used_by
        model_outputs
            id
                shape
                dtype
                is_batched
                created_by
    """
    sorted_model_inputs = sorted(graph.inputs, key=lambda n: n.id)
    input_ids = [f"{graph.input_kw_ids.get(node.id, node.scope)}" for node in sorted_model_inputs]
    input_shapes = [n.tensor_shape for n in sorted_model_inputs]
    input_dtypes = [n.tensor_dtype for n in sorted_model_inputs]
    input_destinations = [list({o.stage_id for o in n.out_edges}) for n in sorted_model_inputs]

    sorted_model_outputs = sorted(graph.outputs, key=lambda n: n.id)
    output_ids = [n.scope for n in sorted_model_outputs]
    output_shapes = [n.tensor_shape for n in sorted_model_outputs]
    output_dtypes = [n.tensor_dtype for n in sorted_model_outputs]
    output_sources = [o.stage_id for o in sorted_model_outputs]

    model_inputs = dict()
    for i, s, d, dsts in zip(input_ids, input_shapes, input_dtypes, input_destinations):
        model_inputs[i] = {"shape": s,
                           "dtype": d,
                           "is_batched": is_batched(s),
                           "used_by": dsts}

    model_outputs = dict()
    for o, s, d, src in zip(output_ids, output_shapes, output_dtypes, output_sources):
        model_outputs[o] = {"shape": s,
                            "dtype": d,
                            "is_batched": is_batched(s),
                            "created_by": src}

    return model_inputs, model_outputs


def generate_switch_batch_size():
    s = """batch_dim = config['batch_dim']
    for d in chain(config['model_inputs'].values(),config['model_outputs'].values()):
        if d['is_batched']:
            shape = d['shape']
            d['shape'] = torch.Size(shape[:batch_dim] + (batch_size,) + shape[batch_dim+1:])
    
    for s in config['stages'].values():
        for d in chain(s['inputs'].values(),s['outputs'].values()):
            if d['is_batched']:
                shape = d['shape']
                d['shape'] = torch.Size(shape[:batch_dim] + (batch_size,) + shape[batch_dim+1:])"""
    return s


def generate_config_without_nested(dict_config: Dict) -> Dict:
    config_without_nested = deepcopy(dict_config)

    # convert in/out for model
    new_model_inputs = dict()
    for input_id, input_cfg in config_without_nested['model_inputs'].items():
        # found nested activation
        if not isinstance(input_cfg['is_batched'], bool):
            flattened_is_batched = flatten(input_cfg['is_batched'])
            flattened_shape = flatten(input_cfg['shape'])
            flattened_dtype = flatten(input_cfg['dtype'])
            for idx, (is_batched, shape, dtype) in enumerate(
                    zip(flattened_is_batched, flattened_shape, flattened_dtype)):
                cfg = {"shape": shape, "dtype": dtype, "is_batched": is_batched, "used_by": input_cfg['used_by']}
                new_model_inputs[input_id + f"_{idx}"] = cfg
        else:
            new_model_inputs[input_id] = input_cfg

    config_without_nested['model_inputs'] = new_model_inputs

    new_model_outputs = dict()
    for output_id, output_cfg in config_without_nested['model_outputs'].items():
        # found nested activation
        if not isinstance(output_cfg['is_batched'], bool):
            flattened_is_batched = flatten(output_cfg['is_batched'])
            flattened_shape = flatten(output_cfg['shape'])
            flattened_dtype = flatten(output_cfg['dtype'])
            for idx, (is_batched, shape, dtype) in enumerate(
                    zip(flattened_is_batched, flattened_shape, flattened_dtype)):
                cfg = {"shape": shape, "dtype": dtype, "is_batched": is_batched, "created_by": output_cfg['created_by']}
                new_model_outputs[output_id + f"_{idx}"] = cfg
        else:
            new_model_outputs[output_id] = output_cfg

    config_without_nested['model_outputs'] = new_model_outputs

    # convert in/out for stages
    for stage_id, stage in config_without_nested['stages'].items():
        new_stage_outputs = dict()
        for output_id, output_cfg in stage['outputs'].items():
            # found nested activation
            if not isinstance(output_cfg['is_batched'], bool):
                flattened_is_batched = flatten(output_cfg['is_batched'])
                flattened_shape = flatten(output_cfg['shape'])
                flattened_dtype = flatten(output_cfg['dtype'])
                flatten_req_grad = flatten(output_cfg['req_grad'])
                for idx, (is_batched, shape, dtype, req_grad) in enumerate(
                        zip(flattened_is_batched, flattened_shape, flattened_dtype, flatten_req_grad)):
                    cfg = {"shape": shape, "dtype": dtype, "req_grad": req_grad, "is_batched": is_batched,
                           "used_by": output_cfg['used_by']}
                    new_stage_outputs[output_id + f"_{idx}"] = cfg
            else:
                new_stage_outputs[output_id] = output_cfg

        stage['outputs'] = new_stage_outputs

        new_stage_inputs = dict()
        for input_id, input_cfg in stage['inputs'].items():
            # found nested activation
            if not isinstance(input_cfg['is_batched'], bool):
                flattened_is_batched = flatten(input_cfg['is_batched'])
                flattened_shape = flatten(input_cfg['shape'])
                flattened_dtype = flatten(input_cfg['dtype'])
                flatten_req_grad = flatten(input_cfg['req_grad'])
                for idx, (is_batched, shape, dtype, req_grad) in enumerate(
                        zip(flattened_is_batched, flattened_shape, flattened_dtype, flatten_req_grad)):
                    cfg = {"shape": shape, "dtype": dtype, "req_grad": req_grad, "is_batched": is_batched,
                           "created_by": input_cfg['created_by']}
                    new_stage_inputs[input_id + f"_{idx}"] = cfg
            else:
                new_stage_inputs[input_id] = input_cfg
        stage['inputs'] = new_stage_inputs

    return config_without_nested


def generate_config_with_input_propagation(dict_config: Dict) -> Dict:
    # consider the case where stage 0 sends an activation for stages[1,2,3]
    # if stage0 sends directly to all other stages
    # it will need to keep the buffer alive until stage3 uses it
    # but if we instead have 0->1->2->3
    # than each stage needs to keep the buffer alive for 1 cycle
    # removing the overall memory used per worker

    new_config = deepcopy(dict_config)

    # modify model outputs
    new_model_outputs = new_config['model_outputs']
    for name, cfg in dict_config['model_outputs'].items():
        old_src = cfg['created_by']
        used_by = dict_config['stages'][old_src]['outputs'][name]['used_by']
        if len(used_by) > 1:
            new_src = max(used_by)
            new_model_outputs[name]['created_by'] = new_src
        else:
            # preserve order
            new_model_outputs[name] = new_model_outputs.pop(name)

    # modify stages
    for stage_id, stage_cfg in dict_config['stages'].items():
        new_stage_cfg = new_config['stages'][stage_id]
        for name, cfg in stage_cfg['inputs'].items():
            old_src = cfg['created_by']
            # model inputs remain unchanged
            if old_src == -1:
                # need to preserve order
                new_stage_cfg['inputs'][name] = new_stage_cfg['inputs'].pop(name)
                continue
            used_by = dict_config['stages'][old_src]['outputs'][name]['used_by']
            if len(used_by) > 1:
                new_src = max((dst for dst in used_by if dst < stage_id), default=old_src)
                new_stage_cfg['inputs'][name]['created_by'] = new_src
                new_stage_cfg['inputs'][name + f"_{stage_id}"] = new_stage_cfg['inputs'].pop(name)
            else:
                # preserve order
                new_stage_cfg['inputs'][name] = new_stage_cfg['inputs'].pop(name)

        for name, cfg in stage_cfg['outputs'].items():
            if len(cfg['used_by']) > 1:
                new_dst = min((dst for dst in cfg['used_by'] if dst > stage_id))
                new_stage_cfg['outputs'][name]['used_by'] = [new_dst]
                new_stage_cfg['outputs'][name + f"_{new_dst}"] = new_stage_cfg['outputs'].pop(name)
            else:
                new_stage_cfg['outputs'][name] = new_stage_cfg['outputs'].pop(name)

    return new_config
