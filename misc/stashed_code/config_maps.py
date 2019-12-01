def get_global_maps(num_stages, mode='seq'):
    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }

    if mode == 'seq':
        configuration_maps = {
            "module_to_stage_map": list(range(num_stages)),
            "stage_to_rank_map": {i: [i] for i in range(num_stages)},
            "stage_to_depth_map": {i: [num_stages - i - 1] for i in range(num_stages)}
        }
    elif mode == 'file':
        raise NotImplementedError()  # TODO
    else:
        raise NotImplementedError()

    module_to_stage_map = configuration_maps['module_to_stage_map']
    stage_to_rank_map = configuration_maps['stage_to_rank_map']
    stage_to_depth_map = configuration_maps['stage_to_depth_map']

    rank_to_stage_map = {}
    # Reverse
    for stage in stage_to_rank_map:
        for rank in stage_to_rank_map[stage]:
            rank_to_stage_map[rank] = stage

    return configuration_maps, rank_to_stage_map
