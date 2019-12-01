# DEPRECATED

from typing import Dict


def create_maps_sequential(num_stages):
    maps = {
        "module_to_stage_map": list(range(num_stages)),
        "stage_to_rank_map": {i: [i] for i in range(num_stages)},
        "stage_to_depth_map": {i: [num_stages - i - 1] for i in range(num_stages)}
    }
    return maps


def get_rank_maps(configs: Dict, mode='no_dp'):
    """ Maps auto-generated configs to ranks, so it could be used with pipedream and `torch.distributed` """

    input_names = configs.pop('model inputs')
    output_names = configs.pop('model outputs')

    # Sanity check.
    for i in configs.keys():
        assert isinstance(i, int)

    num_stages = len(configs)

    # Assuming single rank per stage.
    # TODO: support multiple ranks (data parallelism) per stage
    assert(mode == 'no_dp')
    num_ranks = num_stages
    maps = create_maps_sequential(num_stages)

    return num_stages, num_ranks, maps
