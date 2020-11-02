from copy import deepcopy
from typing import Dict, List, Tuple, Callable

import torch
from torch.nn import Module

from autopipe.autopipe.compiler.utils import pretty_format_obj
from autopipe.autopipe.model_profiling import Graph
from autopipe.autopipe.utils import nested_map, flatten

tab = '    '
dtab = tab + tab
GET_STAGES_ON_CPU_NAME = "DEBUG"


def create_pipeline_configuration(graph: Graph,
                                  ios: Dict[int,
                                            Dict[str,
                                                 List[str]]],
                                  model_blocks: Dict[str, Module],
                                  batch_dim: int,
                                  generate_activation_propagation: bool) -> Tuple[str, Dict]:
    """Generates the create_pipeline_configuration method which given a model creates his partitioned counterpart
    """
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
        f"def create_pipeline_configuration({GET_STAGES_ON_CPU_NAME}=False, batch_size={batch_size}):"
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
        # modify the config to accommodate input propagation
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


def create_stages_config(ios: Dict, is_batched: Callable[[torch.Size], bool], stage_to_device_map=None) -> Dict:
    """generates the stages portion of the config
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
    """
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
        stage_depth = io['depth']

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
                       "devices": f"['cpu' if {GET_STAGES_ON_CPU_NAME} else 'cuda:{idx}']",
                       "stage_depth": stage_depth}
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
