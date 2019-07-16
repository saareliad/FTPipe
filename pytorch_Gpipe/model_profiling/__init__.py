from .control_flow_graph import Graph, NodeTypes
from .optimize_graph import optimize_graph
from .network_profiler import profileNetwork
from ..utils import traverse_model, traverse_params_buffs
from typing import Optional, List, Dict, Any
import torch.nn as nn
import torch

__all__ = ['graph_builder', 'profileNetwork']


def graph_builder(model: nn.Module, *sample_batch, max_depth: int = 1000, weights: Optional[Dict[str, Any]] = None, basic_blocks: Optional[List[nn.Module]] = None, use_profiler=False) -> Graph:
    '''
    returns a graph that models the control flow of the given network by tracing it's forward pass

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    weights:
        an optional dictionary from scopes to Node weights
    use_profiler:
        wether to use weights given by our profiler
        this option supersedes the wieghts option defaults to False
    '''
    weights = weights if weights != None else {}

    if use_profiler:
        weights = profileNetwork(model, *sample_batch, max_depth=max_depth,
                                 basic_blocks=basic_blocks)

    buffer_param_names = map(lambda t: t[1], traverse_params_buffs(model))
    buffer_param_names = list(buffer_param_names)

    layerNames = map(lambda t: t[1], traverse_model(
        model, depth=max_depth, basic_blocks=basic_blocks))
    layerNames = list(layerNames)

    # trace the model and build a graph
    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(
            model, sample_batch)
        trace_graph = trace_graph.graph()

    num_inputs = len(sample_batch)

    graph = Graph(layerNames, num_inputs, buffer_param_names,
                  trace_graph, weights, basic_blocks, max_depth)
    optimize_graph(graph)

    return graph
