from .model_profiling import visualize, visualize_with_profiler, profileNetwork
from .model_partitioning import partition_graph, distribute_model, distribute_model_from_config, sequential_partition
import torch
from typing import Optional, Callable, Any

__all__ = ['partition_with_profiler', 'distribute_using_profiler', 'distribute_using_custom_weights',
           'visualize', 'visualize_with_profiler', 'partition_graph', 'distribute_model', 'distribute_model_from_config']


def partition_with_profiler(model, *sample_batch, nparts=4, max_depth=100, basic_blocks=None, weighting_function: Optional[Callable[[Any], int]] = None):
    graph = visualize_with_profiler(model, *sample_batch, max_depth=max_depth,
                                    basic_blocks=basic_blocks)

    graph, _, _ = partition_graph(
        graph, nparts, weighting_function=weighting_function)

    return graph


def distribute_using_profiler(model, *sample_batch, device_list=None, max_depth=100, basic_blocks=None, return_config=False, weighting_function: Optional[Callable[[Any], int]] = None):
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


def distribute_using_custom_weights(model, weights, *sample_batch, device_list=None, max_depth=100, basic_blocks=None, return_config=False, weighting_function: Optional[Callable[[Any], int]] = None):
    if device_list is None:
        if torch.cuda.is_available():
            device_list = list(range(torch.cuda.device_count()))
        else:
            device_list = ["cpu"]

    graph = visualize(model, *sample_batch, max_depth=max_depth,
                      basic_blocks=basic_blocks, weights=weights)

    graph, _, _ = partition_graph(graph, len(
        device_list), weighting_function=weighting_function if weighting_function != None else lambda w: w)

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
