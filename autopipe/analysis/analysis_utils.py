from collections import deque
from typing import Dict

import torch
from torch import Tensor

from autopipe.autopipe.utils import move_tensors
from pipe.models.simple_partitioning_config import PipelineConfig


class AnalysisPipelineConfig(PipelineConfig):
    def __init__(self, d, layers, tensors):
        super().__init__(d)

        self.stage_to_model = {
            stage_id: self.realize_stage(layers, tensors, stage_id, device='cpu')
            for stage_id in range(self.n_stages)
        }

    def model_inputs(self):
        return self.d['model_inputs']

    def model_outputs(self):
        return self.d['model_outputs']

    def get_inputs_req_grad_for_stage_tuple(self, stage_id: int):
        my_inputs = self.d['stages'][stage_id]['inputs']
        if 'req_grad' in next(iter(my_inputs.values())):
            return tuple(v['req_grad'] for i, v in my_inputs.items())
        else:
            raise NotImplementedError()

    def get_all_stage_inputs(self, stage_id):
        return self.d['stages'][stage_id]['inputs']

    def get_all_stage_outputs(self, stage_id):
        return self.d['stages'][stage_id]['outputs']


def extra_communication_time_lower_bound(comp_time, comm_time):
    """communication is completely parallel to computation """
    if comp_time >= comm_time:
        return 0
    else:
        return comm_time - comp_time


def extra_communication_time_upper_bound(comp_time, comm_time):
    """communication is completely not parallel to computation """
    return comm_time


def upper_utilization_bound(comp_time, comm_time):
    """communication is completely parallel to computation """
    comm_time = extra_communication_time_lower_bound(comp_time, comm_time)
    return comp_time / (comm_time + comp_time)


def lower_utilization_bound(comp_time, comm_time):
    """communication is completely not parallel to computation """
    comm_time = extra_communication_time_upper_bound(comp_time, comm_time)
    return comp_time / (comm_time + comp_time)


def apply_ratio(upper, lower, ratio):
    return (upper * (1 - ratio)) + (lower * ratio)


def convert_to_analysis_format(config: Dict, layers: Dict[str, torch.nn.Module],
                               tensors: Dict[str, Tensor]) -> AnalysisPipelineConfig:
    """convert a pipeline configuration to format used by the analysis module"""
    return AnalysisPipelineConfig(config, layers, tensors)


def run_partitions(model_inputs, analysis_config: AnalysisPipelineConfig,
                   device='cuda'
                   ):
    if not torch.cuda.is_available():
        device = 'cpu'
    # kwarg input
    if isinstance(model_inputs, dict):
        model_inputs = tuple(
            [model_inputs[i] for i in analysis_config.model_inputs()])
    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs,)
    n_partitions = analysis_config.n_stages
    activations = {}
    for i in range(n_partitions):
        analysis_config.stage_to_model[i] = analysis_config.stage_to_model[i].to(device)
        analysis_config.stage_to_model[i].device = device

    for i, t in zip(analysis_config.model_inputs(), model_inputs):
        activations[i] = move_tensors(t, device)

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        # if all inputs are ready run partition
        if all(tensor in activations
               for tensor in analysis_config.get_all_stage_inputs(idx)):
            inputs = [
                activations[tensor]
                for tensor in analysis_config.get_all_stage_inputs(idx)
            ]
            outs = analysis_config.stage_to_model[idx](*inputs)
            for o, t in zip(analysis_config.get_all_stage_outputs(idx), outs):
                activations[o] = t
        else:
            parts.append(idx)

    return [activations[o] for o in analysis_config.model_outputs()]


def add_dicts(d1, d2):
    assert len(d1) == len(d2)
    d = {}
    for (i1, v1), (i2, v2) in zip(d1.items(), d2.items()):
        assert i1 == i2
        d[i1] = v1 + v2
    return d