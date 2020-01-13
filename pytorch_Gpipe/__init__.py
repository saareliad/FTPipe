from typing import Any, Callable, List, Dict, Optional

import torch
import torch.nn as nn

from .model_partitioning import partition
from .module_generation import generatePartitionModules
from .model_profiling import Graph, graph_builder, profileNetwork
from .pipeline import Pipeline
from .utils import Devices, Tensors

__all__ = ['pipe_model', 'partition_with_profiler',
           'partition', 'Pipeline']


# TODO pytorch jit trace / get_trace_graph do not support kwargs
# TODO support basic blocks


def pipe_model(model: nn.Module, sample_batch: Tensors, kwargs: Optional[Dict] = None, n_iter=10, nparts: int = 4,
               depth=1000, basic_blocks: Optional[List[nn.Module]] = None, use_jit_trace=False,
               partition_by_memory: bool = False, weighting_function=None, output_file: str = None, DEBUG=False, **METIS_opt):
    '''attemps to partition a model to given number of parts using our profiler
       this will produce a python file with the partition config

    the generated python file exposes a method named {modelClass}Pipeline that creates the pipeline
    for this specific model config

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    kwargs:
        aditional kwargs dictionary to pass to the model
    n_iter:
        number of profiling iteration used to gather statistics
    nparts:
        the number of partitions
    depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    use_jit_trace:
        wether to use jit.trace() or jit._get_trace_graph() in order to get the models trace
    partition_by_memory:
        whether to partition by memory consumption if False partitions by time defaults to False
    weighting_function:
        an optional function from node weights to non negative integers if not provided a default function will be used
    output_file:
        the file name in which to save the partition config
        if not given defualts to generated_{modelClass}{actualNumberOfPartitions}
    DEBUG:
        whether to generate the debug version of the partition more comments and assertions in the generated file
    METIS_opt:
        additional kwargs to pass to the METIS partitioning algorithm
    '''
    def by_time(w):
        if hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
            return max(int(100 * (w.forward_time + w.backward_time) / 2), 1)
        return 0

    def by_memory(w):
        if hasattr(w, 'cuda_memory_forward') and hasattr(w, 'cuda_memory_backward'):
            return max(int(100 * (w.cuda_memory_forward + w.cuda_memory_backward) / 2), 1)
        return 1

    if weighting_function != None:
        w_func = weighting_function
    elif partition_by_memory:
        w_func = by_memory
    else:
        w_func = by_time

    graph = partition_with_profiler(model, sample_batch, kwargs=kwargs, max_depth=depth, n_iter=n_iter, nparts=nparts,
                                    basic_blocks=basic_blocks, use_jit_trace=use_jit_trace, weighting_function=w_func, METIS_opt=METIS_opt)

    generatePartitionModules(graph, model,
                             output_file=output_file, verbose=DEBUG)

    return graph


def partition_with_profiler(model: nn.Module, sample_batch: Tensors, kwargs: Optional[Dict] = None, n_iter=10, nparts=4, max_depth=100, basic_blocks: Optional[List[nn.Module]] = None,
                            use_jit_trace=False, weighting_function: Optional[Callable[[Any], int]] = None, **METIS_opt) -> Graph:
    '''
    return a graph representing the partitioned model with the weights given by the profiler
    this method does not distribute the model accross devices

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    kwargs:
        aditional kwargs dictionary to pass to the model
    n_iter:
        number of profiling iteration used to gather statistics
    nparts:
        the number of partitions
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    use_jit_trace:
        wether to use jit.trace() or jit._get_trace_graph() in order to get the models trace
    weighting_function:
        an optional function from node weights to non negative integers if not provided a default function will be used
    METIS_opt:
        additional kwargs to pass to the METIS partitioning algorithm
    '''
    graph = graph_builder(model, sample_batch, kwargs=kwargs, max_depth=max_depth,
                          basic_blocks=basic_blocks, n_iter=n_iter, use_profiler=True, use_jit_trace=use_jit_trace)

    graph = partition(graph, nparts, weighting_function=weighting_function,
                      **METIS_opt)

    return graph
