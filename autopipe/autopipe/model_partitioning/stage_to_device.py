from collections import defaultdict
from typing import List

from autopipe.autopipe.model_profiling import NodeTypes


def gen_stage_to_device_map(graph) -> List[int]:
    """
    Args:
        graph:

    Returns:
        l[i] = k => stage i is on device k

    # HACKY lazy code copy pasta
    # TODO: can also calculate number of dummy stages per GPU.
    """
    ######### FROM PREPARE ANALAYSIS KWARGS
    analysis_kwargs = {}
    gpu_to_stages = defaultdict(set)
    stage_to_gpu = dict()
    for n in graph.non_input_nodes:  # we do note care about constants
        if n.gpu_id is None or n.type == NodeTypes.CONSTANT:
            continue
        gpu_to_stages[n.gpu_id].add(n.stage_id)
        if n.stage_id in stage_to_gpu:
            assert stage_to_gpu[n.stage_id] == n.gpu_id, (stage_to_gpu[n.stage_id], n.gpu_id)
        else:
            assert n.gpu_id is not None
        stage_to_gpu[n.stage_id] = n.gpu_id
    if gpu_to_stages:
        analysis_kwargs['stages_on_same_gpu'] = list(gpu_to_stages.values())
    else:
        raise RuntimeError("no GPUs.")


    stage_to_gpu = [stage_to_gpu[i] for i in sorted(stage_to_gpu.keys())]
    # print("stage_to_gpu", stage_to_gpu)  # so why was it not sorted?

    ###### from analysis
    stages_on_same_gpu = analysis_kwargs['stages_on_same_gpu']
    unique_stages_on_same_gpu = stages_on_same_gpu
    stages_on_same_gpu = defaultdict(set)
    for i in unique_stages_on_same_gpu:
        for j in i:
            stages_on_same_gpu[j] = i

    for i in unique_stages_on_same_gpu:
        assert len(i) >= 1

    # num_dummy_stages = sum((len(i) - 1) for i in unique_stages_on_same_gpu)
    # # n_partitions = config.n_stages
    n_partitions = graph.num_partitions

    pipeline_representation_stage_to_device_map = sorted_stage_to_device_map(n_partitions, stages_on_same_gpu)
    return pipeline_representation_stage_to_device_map


def sorted_stage_to_device_map(n_partitions, stages_on_same_gpu):
    pipeline_representation_stage_to_device_map = list()
    for stage_id in range(n_partitions):
        seen_devices = set()
        if stage_id in stages_on_same_gpu:
            device_id = min(stages_on_same_gpu[stage_id])
        else:
            device_id = len(seen_devices)
        seen_devices.add(device_id)
        pipeline_representation_stage_to_device_map.append(device_id)
    # Canonize
    tmp = sorted(set(pipeline_representation_stage_to_device_map))
    tmp = {v: i for i, v in enumerate(tmp)}
    pipeline_representation_stage_to_device_map = [tmp[i] for i in pipeline_representation_stage_to_device_map]
    return pipeline_representation_stage_to_device_map