from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils import Tensors, traverse_model, traverse_params_buffs, model_scopes, _count_elements
from .control_flow_graph import Graph, NodeTypes
from .network_profiler import profileNetwork

__all__ = ['graph_builder', 'profileNetwork']


def graph_builder(model: nn.Module, sample_batch: Tensors = (), kwargs: Optional[Dict] = None, max_depth: int = 1000,
                  basic_blocks: Optional[List[nn.Module]] = None, use_profiler=False, n_iter=1, use_jit_trace=True) -> Graph:
    '''
    returns a graph that models the control flow of the given network by tracing it's forward pass

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    kwargs:
        keyword args to pass to the model
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    use_profiler:
        wether to use weights given by our profiler
        this option supersedes the wieghts option defaults to False
    n_iter:
        the number of iterations used by the profiler to profile the network used only when use_profiler is true
    use_jit_trace:
        wether to use jit.trace() or jit._get_trace_graph() in order to get the models trace
    '''
    weights = dict()
    if kwargs is None:
        kwargs = dict()
    if not isinstance(sample_batch, tuple):
        sample_batch = (sample_batch,)

    if use_profiler:
        weights = profileNetwork(model, sample_batch, kwargs=kwargs, n_iter=n_iter, max_depth=max_depth,
                                 basic_blocks=basic_blocks)

    buffer_param_names = map(lambda t: t[1], traverse_params_buffs(model))
    buffer_param_names = list(buffer_param_names)

    layerNames = model_scopes(model, depth=max_depth,
                              basic_blocks=basic_blocks)
    layerNames = list(layerNames)

    # trace the model and build a graph
    with torch.no_grad():
        if use_jit_trace:
            trace_graph = torch.jit.trace(model, sample_batch,
                                          check_trace=False).graph
        else:
            if hasattr(torch.jit, "get_trace_graph"):
                get_trace_graph = torch.jit.get_trace_graph
            else:
                assert hasattr(torch.jit, "_get_trace_graph")
                get_trace_graph = torch.jit._get_trace_graph
            trace_graph, _ = get_trace_graph(model, sample_batch, kwargs)
            trace_graph = trace_graph.graph()

    num_inputs = _count_elements(*sample_batch) + len(kwargs)

    graph = Graph(layerNames, num_inputs, buffer_param_names,
                  trace_graph, weights, max_depth, use_jit_trace=use_jit_trace)

    return graph
