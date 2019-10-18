from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from .model_partitioning import distribute_model, partition_graph
from .model_profiling import Graph, graph_builder, profileNetwork
from .pipeline import PipelineParallel
from .utils import Devices, Tensors

__all__ = ['pipe_model', 'partition_with_profiler', 'distribute_using_profiler', 'distribute_using_custom_weights',
           'partition_graph', 'distribute_model', 'distribute_by_memory', 'distribute_by_time']


def pipe_model(model: nn.Module, microbatch_size: int, sample_batch: Tensors, devices: Optional[Devices] = None, by_memory: bool = False, depth: int = 100, optimize_pipeline_wrappers: bool = True, return_graph: bool = False) -> PipelineParallel:
    partition_policy = distribute_by_time
    if by_memory:
        partition_policy = distribute_by_memory

    modified_model, wrappers, counter, graph = partition_policy(model, sample_batch,
                                                                devices=devices, depth=depth, optimize_pipeline_wrappers=optimize_pipeline_wrappers)
    in_shape = sample_batch.shape[1:]
    in_shape = tuple(in_shape)

    if not return_graph:
        graph = None

    pipe = PipelineParallel(
        modified_model, microbatch_size, in_shape, wrappers, counter, graph=graph)

    return pipe


def distribute_by_time(model: nn.Module, *sample_batch: Tensors, devices: Optional[Devices] = None, depth: int = 100, optimize_pipeline_wrappers: bool = True):
    '''
    distirbutes a model according to layer's execution time.\n
    this method is a convenience method as is equivalent to:
    distribute_using_profiler(model: nn.Module, *sample_batch, devices, max_depth=100, basic_blocks=None, return_config, weighting_function: Optional[Callable[[Any], int]] =w_func)\n
    where w_func is 100*(layer.forward_time+layer.backward_time)/2

    Parameters:
    -----------
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    devices:
        the devices to distribute the model accross each device will hold a partition
    depth:
        how far down we go in the model tree determines the detail level of the graph    
    optimize_pipeline_wrappers:
        whether to attempt to minimize the number of wrappers inserted by us defualt=True
        you will possibly want to set this to false if your layers have tuple input/output

    '''
    def w_function(w):
        if hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
            return max(int(100 * (w.forward_time + w.backward_time) / 2), 1)
        return 1

    return distribute_using_profiler(model, *sample_batch, devices=devices, weighting_function=w_function, max_depth=depth, optimize_pipeline_wrappers=optimize_pipeline_wrappers)


def distribute_by_memory(model: nn.Module, *sample_batch: Tensors, devices: Optional[Devices] = None, depth=100, optimize_pipeline_wrappers: bool = True):
    '''
    distirbutes a model according to layer's peak memory consumption as recoreded by CUDA in GB.\n
    this method is a convenience method as is equivalent to:
    distribute_using_profiler(model: nn.Module, *sample_batch, devices, max_depth=100, basic_blocks=None, return_config, weighting_function: Optional[Callable[[Any], int]] =w_func)\n
    where w_func is 100*(layer.cuda_memory_forward+layer.cuda_memory_backward)/2

    Parameters:
    -----------
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    devices:
        the devices to distribute the model accross each device will hold a partition
    depth:
        how far down we go in the model tree determines the detail level of the graph
    optimize_pipeline_wrappers:
        whether to attempt to minimize the number of wrappers inserted by us defualt=True
        you will possibly want to set this to false if your layers have tuple input/output
    '''
    def w_function(w):
        if hasattr(w, 'cuda_memory_forward') and hasattr(w, 'cuda_memory_backward'):
            return max(int(100 * (w.cuda_memory_forward + w.cuda_memory_backward) / 2), 1)
        return 1

    return distribute_using_profiler(model, *sample_batch, devices=devices, weighting_function=w_function, max_depth=depth, optimize_pipeline_wrappers=optimize_pipeline_wrappers)


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
        an optional function from node weights to non negative integers if not provided a deafualt function will be used
    '''
    graph = graph_builder(model, *sample_batch, max_depth=max_depth,
                          basic_blocks=basic_blocks, use_profiler=True)

    graph = partition_graph(
        graph, nparts, weighting_function=weighting_function)

    return graph


def distribute_using_profiler(model: nn.Module, *sample_batch: Tensors, devices: Optional[Devices] = None, max_depth=100, basic_blocks: Optional[List[nn.Module]] = None, weighting_function: Optional[Callable[[Any], int]] = None, optimize_pipeline_wrappers: bool = True):
    '''
    distributes a model accross the given devices in accordance to given specifications and data from the profiler\n
    !!!this method changes the the given model and is part of the internal API

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    devices:
        the devices to distribute the model accross each device will hold a partition
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    weighting_function:
        an optional function from node weights to non negative integers if not provided a defualt function will be used
    optimize_pipeline_wrappers:
        whether to attempt to minimize the number of wrappers inserted by us defualt=True
        you will possibly want to set this to false if your layers have tuple input/output
    '''

    if devices is None:
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            raise ValueError('CUDA is required but is not available')

    graph = partition_with_profiler(model, *sample_batch, nparts=len(devices),
                                    max_depth=max_depth, basic_blocks=basic_blocks, weighting_function=weighting_function)

    result = distribute_model(model, devices, graph,
                              *sample_batch, optimize_pipeline_wrappers=optimize_pipeline_wrappers)

    return result + (graph,)


def distribute_using_custom_weights(model: nn.Module, weights, *sample_batch: Tensors, devices: Optional[Devices] = None, max_depth=100, basic_blocks: Optional[List[nn.Module]] = None, weighting_function: Optional[Callable[[Any], int]] = None, optimize_pipeline_wrappers: bool = True):
    '''
    distributes a model accross the given devices in accordance to given specifications and custom node weights\n
    !!!this method changes the the given model and is part of the internal API

    Parameters:
    model:
        the network we wish to model
    weights:
        a dictionary from scopeNames to a weight object
    sample_batch:
        a sample input to use for tracing
    devices:
        the devices to distribute the model accross each device will hold a partition
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    weighting_function:
        an optional function from node weights to non negative integers if not provided a defualt function will be used
    optimize_pipeline_wrappers:
        whether to attempt to minimize the number of wrappers inserted by us defualt=True
        you will possibly want to set this to false if your layers have tuple input/output
    '''
    if devices is None:
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = ["cpu"]

    graph = graph_builder(model, *sample_batch, max_depth=max_depth,
                          basic_blocks=basic_blocks, weights=weights)

    graph = partition_graph(graph, len(devices),
                            weighting_function=weighting_function if weighting_function != None else lambda w: w)

    res = distribute_model(model, devices, graph,
                           *sample_batch, optimize_pipeline_wrappers=optimize_pipeline_wrappers)

    return res + (graph,)
