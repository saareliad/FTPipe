from .model_profiling import visualize_with_profiler, profileNetwork, graph_builder
from .model_partitioning import partition_graph, distribute_model, distribute_model_from_config, sequential_partition
from .pipeline import PipelineParallel
import torch
import torch.nn as nn
from typing import Optional, Callable, Any

__all__ = ['pipe_model', 'partition_with_profiler', 'distribute_using_profiler', 'distribute_using_custom_weights',
           'visualize_with_profiler', 'partition_graph', 'distribute_model', 'distribute_model_from_config', 'distribute_by_memory', 'distribute_by_time']


def pipe_model(model: nn.Module, microbatch_size, sample_batch, device_list=None):
    modified_model, wrappers, counter, _ = distribute_by_time(
        model, sample_batch, device_list=device_list)

    in_shape = []
    if isinstance(sample_batch, torch.Tensor):
        in_shape.append(sample_batch.shape[1:])
    else:
        for t in sample_batch:
            in_shape.append(t.shape[1:])

    in_shape = tuple(in_shape)

    pipe = PipelineParallel(
        modified_model, microbatch_size, in_shape, wrappers, counter)

    return pipe


def distribute_by_time(model: nn.Module, *sample_batch, device_list=None, return_config=False):
    '''
    distirbutes a model according to layer's execution time.\n
    this method is a convenience method as is equivalent to:
    distribute_using_profiler(model: nn.Module, *sample_batch, device_list, max_depth=100, basic_blocks=None, return_config, weighting_function: Optional[Callable[[Any], int]] =w_func)\n
    where w_func is 100*(layer.forward_time+layer.backward_time)/2

    Parameters:
    -----------
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    device_list:
        the devices to distribute the model accross each device will hold a partition
    return_config:
        wheter to return a configuration of the partition useful if you wish to do the partitioning process once
        and use it for multiple instances of the model
    '''
    def w_function(w):
        if isinstance(w, tuple) and hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
            return max(int(100*(w.forward_time+w.backward_time)/2), 1)
        return 1

    return distribute_using_profiler(model, *sample_batch, device_list=device_list, return_config=return_config, weighting_function=w_function)


def distribute_by_memory(model: nn.Module, *sample_batch, device_list=None, return_config=False):
    '''
    distirbutes a model according to layer's peak memory consumption as recoreded by CUDA in GB.\n
    this method is a convenience method as is equivalent to:
    distribute_using_profiler(model: nn.Module, *sample_batch, device_list, max_depth=100, basic_blocks=None, return_config, weighting_function: Optional[Callable[[Any], int]] =w_func)\n
    where w_func is 100*(layer.cuda_memory_forward+layer.cuda_memory_backward)/2

    Parameters:
    -----------
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    device_list:
        the devices to distribute the model accross each device will hold a partition
    return_config:
        wheter to return a configuration of the partition useful if you wish to do the partitioning process once
        and use it for multiple instances of the model
    '''
    def w_function(w):
        if hasattr(w, 'cuda_memory_forward') and hasattr(w, 'cuda_memory_backward'):
            return max(int(100*(w.cuda_memory_forward+w.cuda_memory_backward)/2), 1)
        return 1

    return distribute_using_profiler(model, *sample_batch, device_list=device_list, return_config=return_config, weighting_function=w_function)


def partition_with_profiler(model: nn.Module, *sample_batch, nparts=4, max_depth=100, basic_blocks=None, weighting_function: Optional[Callable[[Any], int]] = None):
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
    basic_block:
        an optional list of modules that if encountered will not be broken down
    weighting_function:
        an optional function from node weights to non negative integers if not provided a deafualt function will be used
    '''
    graph = visualize_with_profiler(model, *sample_batch, max_depth=max_depth,
                                    basic_blocks=basic_blocks)

    graph, _ = partition_graph(
        graph, nparts, weighting_function=weighting_function)

    return graph


def distribute_using_profiler(model: nn.Module, *sample_batch, device_list=None, max_depth=100, basic_blocks=None, return_config=False, weighting_function: Optional[Callable[[Any], int]] = None):
    '''
    distributes a model accross the given devices in accordance to given specifications and data from the profiler\n
    !!!this method changes the the given model and is part of the internal API

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    device_list:
        the devices to distribute the model accross each device will hold a partition
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_block:
        an optional list of modules that if encountered will not be broken down
    return_config:
        wheter to return a configuration of the partition useful if you wish to do the partitioning process once
        and use it for multiple instances of the model
    weighting_function:
        an optional function from node weights to non negative integers if not provided a defualt function will be used
    '''

    if device_list is None:
        if torch.cuda.is_available():
            device_list = list(range(torch.cuda.device_count()))
        else:
            raise ValueError('CUDA is required but is not available')

    graph = partition_with_profiler(model, *sample_batch, nparts=len(device_list),
                                    max_depth=max_depth, basic_blocks=basic_blocks, weighting_function=weighting_function)

    result = distribute_model(model, device_list, graph,
                              *sample_batch, return_config=return_config)

    return result+(graph,)


def distribute_using_custom_weights(model: nn.Module, weights, *sample_batch, device_list=None, max_depth=100, basic_blocks=None, return_config=False, weighting_function: Optional[Callable[[Any], int]] = None):
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
    device_list:
        the devices to distribute the model accross each device will hold a partition
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_block:
        an optional list of modules that if encountered will not be broken down
    return_config:
        wheter to return a configuration of the partition useful if you wish to do the partitioning process once
        and use it for multiple instances of the model
    weighting_function:
        an optional function from node weights to non negative integers if not provided a defualt function will be used
    '''
    if device_list is None:
        if torch.cuda.is_available():
            device_list = list(range(torch.cuda.device_count()))
        else:
            device_list = ["cpu"]

    graph = graph_builder(model, *sample_batch, max_depth=max_depth,
                          basic_block=basic_blocks, weights=weights)

    graph, _, = partition_graph(graph, len(device_list),
                                weighting_function=weighting_function if weighting_function != None else lambda w: w)

    res = distribute_model(model, device_list, graph,
                           *sample_batch, return_config=return_config)

    return res+(graph,)
