from .model_profiling import visualize, visualize_with_profiler
from .model_partitioning import partition_graph, wrap_and_move
import torch

__all__ = ['partition_with_profiler', 'distribute_using_profiler', 'distribute_using_custom_weights',
           'visualize', 'visualize_with_profiler', 'partition_graph', 'wrap_and_move']


def partition_with_profiler(model, *sample_batch, nparts=4, num_iter=4, max_depth=100, basic_blocks=None):
    graph = visualize_with_profiler(model, *sample_batch, max_depth=max_depth,
                                    basic_blocks=basic_blocks, num_iter=num_iter)

    graph, _, _ = partition_graph(graph, nparts)

    return graph


def distribute_using_profiler(model, *sample_batch, device_list=None, num_iter=4, max_depth=100, basic_blocks=None):
    if device_list is None:
        if torch.cuda.is_available():
            device_list = list(range(torch.cuda.device_count()))
        else:
            device_list = ["cpu"]

    graph = partition_with_profiler(model, *sample_batch, nparts=len(device_list),
                                    num_iter=num_iter, max_depth=max_depth, basic_blocks=basic_blocks)

    modified_model, wrappers, counter = wrap_and_move(
        model, basic_blocks, device_list, graph, *sample_batch)

    return modified_model, graph, (counter, wrappers, sample_batch)


def distribute_using_custom_weights(model, weights, *sample_batch, device_list=None, max_depth=100, basic_blocks=None):
    if device_list is None:
        if torch.cuda.is_available():
            device_list = list(range(torch.cuda.device_count()))
        else:
            device_list = ["cpu"]

    graph = visualize(model, *sample_batch, max_depth=max_depth,
                      basic_blocks=basic_blocks, weights=weights)

    graph, _, _ = partition_graph(graph, len(
        device_list), weighting_function=lambda w: w)

    modified_model, wrappers, counter = wrap_and_move(
        model, basic_blocks, device_list, graph, *sample_batch)

    return modified_model, graph, (counter, wrappers, sample_batch)
