from .network_profiler import profileNetwork
from .graph import partition_model


def partition_network_using_profiler(model, num_gpus, *sample_batch, num_iter=4, max_depth=100, basic_blocks=None, device="cuda"):
    weights = profileNetwork(model, *sample_batch, max_depth=max_depth,
                             basic_block=basic_blocks, device=device, num_iter=num_iter)

    return partition_model(model, num_gpus, *sample_batch, num_iter=num_iter, max_depth=max_depth,
                           basic_blocks=basic_blocks, device=device, weights=weights)
