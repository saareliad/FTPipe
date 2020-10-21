import sys
sys.path.append("../")
from typing import Dict
import torch
from torch import Tensor
from collections import deque
from autopipe.utils import move_tensors

def extra_communication_time_lower_bound(comp_time, comm_time):
    """communication is completly parallel to computation """
    if comp_time >= comm_time:
        return 0
    else:
        return comm_time - comp_time


def extra_communication_time_upper_bound(comp_time, comm_time):
    """communication is completly not parallel to computation """
    return comm_time


def upper_utilization_bound(comp_time, comm_time):
    """communication is completly parallel to computation """
    comm_time = extra_communication_time_lower_bound(comp_time, comm_time)
    return comp_time / (comm_time + comp_time)


def lower_utilization_bound(comp_time, comm_time):
    """communication is completly not parallel to computation """
    comm_time = extra_communication_time_upper_bound(comp_time, comm_time)
    return comp_time / (comm_time + comp_time)


def apply_ratio(upper, lower, ratio):
    return (upper * (1 - ratio)) + (lower * ratio)


def convert_to_analysis_format(config:Dict, layers: Dict[str, Tensor], tensors: Dict[str, Tensor]) -> Dict:
    """convert a pipeline configuration to a simpler format used by the analysis module
       this is a convertion to a legacy format which should not be used except for the analysis"""
    analysis_config = dict()

    analysis_config['model inputs'] = config['model_inputs']
    analysis_config['model outputs'] = config['model_outputs']

    for idx, cfg in config['stages'].items():
        stage_config = dict()
        stage_config['inputs'] = list(cfg['inputs'].keys())
        stage_config['outputs'] = list(cfg['outputs'].keys())

        stage = cfg['stage_cls'](layers, tensors,device=cfg['devices'][0])
        stage_config['model'] = stage
        analysis_config[idx] = stage_config

    return analysis_config




def run_partitions(model_inputs, analysis_config):
    #kwarg input
    if isinstance(model_inputs, dict):
        model_inputs = tuple(
            [model_inputs[i] for i in analysis_config['model inputs']])

    n_partitions = sum(1 for k in analysis_config if isinstance(k, int))

    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    activations = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(n_partitions):
        analysis_config[i]['model'] = analysis_config[i]['model'].to(device)
        analysis_config[i]['model'].device = device

    for i, t in zip(analysis_config['model inputs'], model_inputs):
        activations[i] = move_tensors(t, device)

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        # if all inputs are ready run partition
        if all(tensor in activations
               for tensor in analysis_config[idx]['inputs']):
            inputs = [
                activations[tensor]
                for tensor in analysis_config[idx]['inputs']
            ]
            outs = analysis_config[idx]['model'](*inputs)
            for o, t in zip(analysis_config[idx]['outputs'], outs):
                activations[o] = t
        else:
            parts.append(idx)

    return [activations[o] for o in analysis_config['model outputs']]

