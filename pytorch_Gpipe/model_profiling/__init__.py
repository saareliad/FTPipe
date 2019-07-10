from .control_flow_graph import Graph, NodeTypes
from .optimize_graph import optimize_graph
from .network_profiler import profileNetwork
from ..utils import *
from typing import Optional, List, Dict, Any
import torch.nn as nn
import torch

__all__ = ['graph_builder', 'visualize_with_profiler', 'profileNetwork']


def visualize_with_profiler(model: nn.Module, *sample_batch, max_depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None) -> Graph:
    '''
    returns a graph that models the control flow of the given model by tracing it's forward pass
    with weights provided by the profiler

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_block:
        an optional list of modules that if encountered will not be broken down
    '''

    layers_profile = profileNetwork(model, *sample_batch, max_depth=max_depth,
                                    basic_block=basic_blocks)

    return graph_builder(model, *sample_batch, max_depth=max_depth, basic_block=basic_blocks, weights=layers_profile)


def graph_builder(model: nn.Module, *sample_batch, max_depth: int = 1000, weights: Optional[Dict[str, Any]] = None, basic_block: Optional[List[nn.Module]] = None) -> Graph:
    '''
    returns a graph that models the control flow of the given model by tracing it's forward pass

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_block:
        an optional list of modules that if encountered will not be broken down
    weights:
        an optional dictionary from scopes to Node weights
    '''
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

    graph = Graph(layerNames, num_inputs, buffer_param_names,
                  trace_graph, weights, basic_block, max_depth)
    optimize_graph(graph)

    return graph
