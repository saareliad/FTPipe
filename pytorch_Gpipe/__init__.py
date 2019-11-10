from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from .model_partitioning import generatePartitionModules, partition_graph, partition_networkx
from .model_profiling import Graph, graph_builder, profileNetwork
from .pipeline import Pipeline
from .utils import Devices, Tensors

__all__ = ['pipe_model', 'partition_with_profiler',
           'partition_graph', 'partition_networkx', 'Pipeline']


def pipe_model(model: nn.Module, *sample_batch: Tensors, nparts: int = 4, partition_by_memory: bool = False, output_file: str = None, DEBUG=False):
    '''attemps to partition a model to given number of parts using our profiler
       this will produce a python file with the partition config

    the generated python file exposes a method named {modelClass}Pipeline that creates the pipeline
    for this specific model config

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    nparts:
        the number of partitions
    partition_by_memory:
        whether to partition by memory consumption if False partitions by time defaults to False
    output_file:
        the file name in which to save the partition config
        if not given defualts to generated_{modelClass}{actualNumberOfPartitions}
    DEBUG:
        whether to generate the debug version of the partition more comments and assertions in the generated file
    '''
    def by_time(w):
        if hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
            return max(int(100 * (w.forward_time + w.backward_time) / 2), 1)
        return 1

    def by_memory(w):
        if hasattr(w, 'cuda_memory_forward') and hasattr(w, 'cuda_memory_backward'):
            return max(int(100 * (w.cuda_memory_forward + w.cuda_memory_backward) / 2), 1)
        return 1

    if partition_by_memory:
        w_func = by_memory
    else:
        w_func = by_time

    graph = partition_with_profiler(model, *sample_batch, nparts=nparts,
                                    weighting_function=w_func)

    generatePartitionModules(graph, model,
                             output_file=output_file, verbose=DEBUG)

    return graph


def partition_with_profiler(model: nn.Module, *sample_batch: Tensors, nparts=4, max_depth=100, basic_blocks: Optional[List[nn.Module]] = None, weighting_function: Optional[Callable[[Any], int]] = None) -> Graph:
    '''
    return a graph representing the partitioned model with the weights given by the profiler
    this method does not distribute the model accross devices

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    nparts:
        the number of partitions
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    weighting_function:
        an optional function from node weights to non negative integers if not provided a default function will be used
    '''
    graph = graph_builder(model, *sample_batch, max_depth=max_depth,
                          basic_blocks=basic_blocks, use_profiler=True)

    graph = partition_networkx(
        graph, nparts, weighting_function=weighting_function)

    return graph
