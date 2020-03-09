from .cfg_to_model import get_partitioning_v3

from itertools import count
from collections import OrderedDict
import numpy as np


def get_my_send_recv_ranks(config, stage, stage_to_rank_map=None):
    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    # TODO: We assume this is same order with Alon's code/config, after poped some stuff.
    # Alon config is outside of the project, this is dangerous programing...
    receive_ranks = OrderedDict()
    send_ranks = OrderedDict()

    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            # Update only for this stage...
            if i != stage and j != stage:
                continue

            stage_i = config[i]
            stage_j = config[j]
            for tensor_name in stage_i['outputs']:
                if tensor_name in stage_j['inputs']:
                    if stage == j:
                        receive_ranks[tensor_name] = ranks_in_stage(i)
                    else:
                        send_ranks[tensor_name] = ranks_in_stage(j)

    return send_ranks, receive_ranks


def tensor_tags_from_config(config,
                            num_chunks=1,
                            target_tensor_names=None,
                            GRAD_UGLY_SHAMEFUL_NAME="_grad"):
    def config_to_tuples_array(config):
        def config_to_tuples_generator(config):
            """ allows iterating with the tuple: (stage_id, inputs, outputs) """
            for i, v in config.items():
                yield i, v['inputs'], v['outputs']

        return np.array(list(config_to_tuples_generator(config)))

    # Note: same tags for all proccess

    tensor_tags = {}
    tensor_tag = 1
    model = config_to_tuples_array(config)

    for (_, input_tensors, output_tensors) in model:
        for input_tensor in input_tensors:
            if input_tensor not in tensor_tags:
                tensor_tags[input_tensor] = tensor_tag
                tensor_tag += num_chunks
        for output_tensor in output_tensors:
            if output_tensor not in tensor_tags:
                tensor_tags[output_tensor] = tensor_tag
                tensor_tag += num_chunks
    # Create different tags for gradients
    for (_, input_tensors, output_tensors) in model:
        for input_tensor in input_tensors:
            input_tensor += GRAD_UGLY_SHAMEFUL_NAME
            if input_tensor not in tensor_tags:
                tensor_tags[input_tensor] = tensor_tag
                tensor_tag += num_chunks
        for output_tensor in output_tensors:
            output_tensor += GRAD_UGLY_SHAMEFUL_NAME
            if output_tensor not in tensor_tags:
                tensor_tags[output_tensor] = tensor_tag
                tensor_tag += num_chunks

    if target_tensor_names:
        for target_tensor_name in sorted(target_tensor_names):
            tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += num_chunks

    # tensor_tags["ack"] = tensor_tag
    tensor_tag += num_chunks

    return tensor_tags, tensor_tag


class PartitioningConfigParser:
    def __init__(self,
                 cfg,
                 rank,
                 bs_train,
                 bs_eval,
                 model_instance=None,
                 send_target_in_pipe=False):

        pipe_config, config, model = get_partitioning_v3(
            cfg, rank, bs_train, model_instance=model_instance)

        self.model = model
        self.num_stages = len(config['stages'])
        self.stage = pipe_config.rank_to_stage_idx(rank)

        counter = count()
        stage_to_rank_map = {
            i: [next(counter) for _ in stage['devices']]
            for i, stage in config['stages'].items()
        }
        self.send_ranks, self.receive_ranks = get_my_send_recv_ranks(
            config['stages'], self.stage, stage_to_rank_map=stage_to_rank_map)

        self.tensor_tags, self.TOTAL_TAGS = tensor_tags_from_config(
            config['stages'])

        if send_target_in_pipe:
            self.target_tensor_names = config['model_outputs']  # else None
            self.ranks_in_previous_stage = stage_to_rank_map.get(
                self.stage - 1) if self.stage > 0 else []
            self.ranks_in_next_stage = stage_to_rank_map.get(
                self.stage + 1) if self.stage < self.num_stages - 1 else []
        else:
            self.target_tensor_names = None
            self.ranks_in_previous_stage = None
            self.ranks_in_next_stage = None

        self.training_tensor_shapes = pipe_config.shapes()
        pipe_config.change_batch(bs_eval, for_replicated=True)
        self.eval_tensor_shapes = pipe_config.shapes()

        # TODO: infer dtypes in partitioning.
        self.training_tensor_dtypes = {
            i: None
            for i in self.training_tensor_shapes
        }
        self.eval_tensor_dtypes = {i: None for i in self.eval_tensor_shapes}

    def comm_init_args(self):
        return (self.receive_ranks, self.send_ranks, self.tensor_tags,
                self.target_tensor_names, self.ranks_in_previous_stage,
                self.ranks_in_next_stage, self.TOTAL_TAGS)


# model
# num_stages
# stage

# Comm handler:

# receive_ranks,
# send_ranks,
# tensor_tags,
# target_tensor_names,
# ranks_in_previous_stage,
# ranks_in_next_stage,
# TOTAL_TAGS

# Partition Manager:

# training_tensor_dtypes,
# training_tensor_shapes,
# eval_tensor_shapes
