from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils import Tensors, traverse_model, tensorDict, model_scopes, _count_elements
from .control_flow_graph import Graph, NodeTypes
from .network_profiler import profileNetwork
from .graph_builder import build_graph

__all__ = ['graph_builder', 'profileNetwork', 'build_graph']


def graph_builder(model: nn.Module, sample_batch: Tensors = (), kwargs: Optional[Dict] = None, max_depth: int = 1000,
                  basic_blocks: Optional[List[nn.Module]] = None, use_profiler=False, n_iter=1, use_jit_trace=False) -> Graph:
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
        # TODO tracing not tested with kwargs
    assert len(kwargs) == 0, "kwargs not supported yet"

    if not isinstance(sample_batch, tuple):
        sample_batch = (sample_batch,)

    if basic_blocks is None:
        basic_blocks = ()
    else:
        basic_blocks = tuple(basic_blocks)

    if use_profiler:
        weights = profileNetwork(model, sample_batch, kwargs=kwargs, n_iter=n_iter, max_depth=max_depth,
                                 basic_blocks=basic_blocks)

    tensors = [(f"input{i}", t.size()) for i, t in enumerate(sample_batch)] +\
              [(k, t.size()) for k, t in tensorDict(model).items()]

    layerNames = model_scopes(model, depth=max_depth,
                              basic_blocks=basic_blocks)
    layerNames = list(layerNames)

    # trace the model and build a graph
    with torch.no_grad():
        if use_jit_trace:
            old_value = torch._C._jit_get_inline_everything_mode()
            torch._C._jit_set_inline_everything_mode(True)
            trace_graph = torch.jit.trace(model, sample_batch,
                                          check_trace=False).graph
            torch._C._jit_set_inline_everything_mode(old_value)
        else:
            if hasattr(torch.jit, "get_trace_graph"):
                get_trace_graph = torch.jit.get_trace_graph
            else:
                assert hasattr(torch.jit, "_get_trace_graph")
                get_trace_graph = torch.jit._get_trace_graph
            trace_graph, _ = get_trace_graph(model, sample_batch, kwargs)
            trace_graph = trace_graph.graph()

    num_inputs = _count_elements(*sample_batch) + len(kwargs)

    graph = Graph(layerNames, num_inputs, tensors,
                  trace_graph, weights, max_depth, basic_blocks, use_jit_trace=use_jit_trace)

    return graph
