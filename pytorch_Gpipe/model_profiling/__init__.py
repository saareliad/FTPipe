from .control_flow_graph import Graph, NodeTypes
from .optimize_graph import optimize_graph
from .network_profiler import profileNetwork
from ..utils import *
from typing import Optional, List, Dict, Any
import torch.nn as nn
import torch

__all__ = ['visualize', 'visualize_with_profiler', 'profileNetwork']


def visualize(model: nn.Module, *sample_batch, max_depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None, weights: Optional[Dict[str, Any]] = None)->Graph:
    graph = graph_builder(
        model, *sample_batch, max_depth=max_depth, basic_block=basic_blocks, weights=weights)

    optimize_graph(graph)

    return graph


def visualize_with_profiler(model: nn.Module, *sample_batch, max_depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None, num_iter: int = 1)->Graph:
    layers_profile = profileNetwork(model, *sample_batch, max_depth=max_depth,
                                    basic_block=basic_blocks, num_iter=num_iter)

    return visualize(model, *sample_batch, max_depth=max_depth, basic_blocks=basic_blocks, weights=layers_profile)


def graph_builder(model: nn.Module, *sample_batch, max_depth: int = 1000, weights: Optional[Dict[str, Any]] = None, basic_block: Optional[List[nn.Module]] = None)->Graph:
    weights = weights if weights != None else {}

    buffer_param_names = map(lambda t: t[1], traverse_params_buffs(model))
    buffer_param_names = list(buffer_param_names)

    layerNames = map(lambda t: t[1], traverse_model(
        model, max_depth, basic_block))
    layerNames = list(layerNames)

    # trace the model and build a graph
    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(
            model, sample_batch)
        trace_graph = trace_graph.graph()

    num_inputs = len(sample_batch)
    return Graph(layerNames, num_inputs, buffer_param_names, trace_graph, weights, basic_block, max_depth)
