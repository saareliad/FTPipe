from model_partition.network_profiler import profileNetwork
from model_partition.graph import partition_model
from model_partition.distribute_to_gpus import wrap_and_move
import torch


def distribute_model(model, *sample_batch, device_list=None, num_iter=4, max_depth=100, basic_blocks=None):
    if device_list is None:
        if torch.cuda.is_available():
            device_list = list(range(torch.cuda.device_count()))
        else:
            device_list = ["cpu"]

    graph, _, _ = partition_network_using_profiler(model, len(device_list), *sample_batch, num_iter=num_iter,
                                                   max_depth=max_depth, basic_blocks=basic_blocks)

    batch = map(lambda t: t.to(device_list[0]), sample_batch)
    distributed_model, wrappers = wrap_and_move(
        model, basic_blocks, device_list, graph, *batch)

    return distributed_model, graph, wrappers


def partition_network_using_profiler(model, num_gpus, *sample_batch, num_iter=4, max_depth=100, basic_blocks=None):
    weights = profileNetwork(model, *sample_batch, max_depth=max_depth,
                             basic_block=basic_blocks, num_iter=num_iter)

    return partition_model(model, num_gpus, *sample_batch, max_depth=max_depth,
                           basic_blocks=basic_blocks, weights=weights)
