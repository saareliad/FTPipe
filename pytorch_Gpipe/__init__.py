from .model_profiling import visualize_with_profiler, profileNetwork, graph_builder
from .model_partitioning import partition_graph, distribute_model, distribute_model_from_config, sequential_partition
import torch
import torch.nn as nn
from typing import Optional, Callable, Any

__all__ = ['partition_with_profiler', 'distribute_using_profiler', 'distribute_using_custom_weights',
           'visualize_with_profiler', 'partition_graph', 'distribute_model', 'distribute_model_from_config']


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

    graph, _, _ = partition_graph(
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

    if return_config:
        modified_model, wrappers, counter, config = distribute_model(model,
                                                                     device_list, graph, *sample_batch, return_config=True)
        return modified_model, graph, (counter, wrappers, sample_batch), config

    modified_model, wrappers, counter = distribute_model(model,
                                                         device_list, graph, *sample_batch, return_config=False)
    return modified_model, graph, (counter, wrappers, sample_batch)


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

    graph, _, _ = partition_graph(graph, len(device_list),
                                  weighting_function=weighting_function if weighting_function != None else lambda w: w)

    if return_config:
        modified_model, wrappers, counter, config = distribute_model(model,
                                                                     device_list, graph, *sample_batch, return_config=True)
        return modified_model, graph, (counter, wrappers, sample_batch), config

    modified_model, wrappers, counter = distribute_model(model,
                                                         device_list, graph, *sample_batch, return_config=False)
    return modified_model, graph, (counter, wrappers, sample_batch)


# TODO partition by time
def distribute_by_time():
    pass


# TODO partition by memory
def distribute_by_memory():
    pass
