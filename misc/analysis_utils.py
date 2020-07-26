from typing import Dict
from torch import Tensor


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