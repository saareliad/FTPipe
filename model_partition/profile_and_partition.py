from model_partition.network_profiler import profileNetwork
from model_partition.graph import partition_model
from model_partition.distribute_to_gpus import move_to_devices


def distribute_model(model, device_list, *sample_batch, num_iter=4, max_depth=100, basic_blocks=None, trace_device="cuda"):
    graph, _, _ = partition_network_using_profiler(model, num_gpus=len(device_list), *sample_batch, num_iter=num_iter,
                                                   max_depth=max_depth, basic_blocks=basic_blocks, device=trace_device)

    distributed_model = move_to_devices(
        model, max_depth, basic_blocks, device_list, graph)

    return distributed_model, graph


def partition_network_using_profiler(model, num_gpus, *sample_batch, num_iter=4, max_depth=100, basic_blocks=None, device="cuda"):
    weights = profileNetwork(model, *sample_batch, max_depth=max_depth,
                             basic_block=basic_blocks, device=device, num_iter=num_iter)

    return partition_model(model, num_gpus, *sample_batch, max_depth=max_depth,
                           basic_blocks=basic_blocks, device=device, weights=weights)
