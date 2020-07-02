from .cfg_to_model import get_partitioning

from itertools import count
from collections import OrderedDict
import numpy as np


def get_my_send_recv_ranks(config, stage, stage_to_rank_map=None):
    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    stages = config.stages
    receive_ranks = OrderedDict()
    send_ranks = OrderedDict()

    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            # Update only for this stage...
            if i != stage and j != stage:
                continue
            stage_i = stages[i]
            stage_j = stages[j]
            for tensor_name in stage_i.outputs:
                if tensor_name in stage_j.inputs:
                    if stage == j:
                        receive_ranks[tensor_name] = ranks_in_stage(i)
                    else:
                        send_ranks[tensor_name] = ranks_in_stage(j)

    # Enusure the sort order is like config.
    send_ranks = OrderedDict(
        (k, send_ranks[k]) for k in stages[stage].outputs if k in send_ranks)
    receive_ranks = OrderedDict((k, receive_ranks[k])
                                for k in stages[stage].inputs
                                if k in receive_ranks)

    return send_ranks, receive_ranks

# FIXME: for tuples
def tensor_tags_from_config(config,
                            num_chunks=1,
                            target_tensor_names=None,
                            GRAD_UGLY_SHAMEFUL_NAME="_grad"):
    def config_to_tuples_array(config):
        def config_to_tuples_generator(stages):
            """ allows iterating with the tuple: (stage_id, inputs, outputs) """
            for i, v in stages.items():
                yield i, v.inputs, v.outputs

        return np.array(list(config_to_tuples_generator(config.stages)))

    # Note: same tags for all process

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
                 send_target_in_pipe=False,
                 stage_to_device_map=None,
                 stateless_tied_same_process=False):

        pipe_config, model = get_partitioning(cfg,
                                              rank,
                                              bs_train,
                                              model_instance=model_instance)

        # Also save config
        self.pipe_config = pipe_config

        self.model = model
        self.num_stages = len(pipe_config.stages)
        self.stage = pipe_config.rank_to_stage_idx(rank)

        counter = count()
        # Create stage_to_rank_map
        # (1) normal
        # (2) tied wieghts, using the same process (And multithreading)
        if not stateless_tied_same_process:
            stage_to_rank_map = {
                i: [next(counter) for _ in stage.devices]
                for i, stage in pipe_config.stages.items()
            }
        else:
            assert stage_to_device_map is not None
            assert len(stage_to_device_map) == len(pipe_config.stages)
            stage_to_rank_map = {}
            record_ranks = {}
            for i, stage in pipe_config.stages.items():
                tied_to = [
                    k for k, j in enumerate(stage_to_device_map)
                    if j == stage_to_device_map[i]
                ]
                if len(tied_to) > 1 and i == min(tied_to):
                    my_ranks = [next(counter) for _ in stage.devices]
                    for k in tied_to:
                        record_ranks[i] = my_ranks
                elif len(tied_to) > 1 and i != min(tied_to):
                    my_ranks = record_ranks[i]
                else:
                    my_ranks = record_ranks[i]

                stage_to_rank_map[i] = my_ranks

        self.send_ranks, self.receive_ranks = get_my_send_recv_ranks(
            pipe_config, self.stage, stage_to_rank_map=stage_to_rank_map)

        tag_info = tensor_tags_from_config(pipe_config)  # FIXME: for tuples. (can avoid None but its minor)
        self.tensor_tags, self.TOTAL_TAGS = tag_info

        if send_target_in_pipe:
            self.target_tensor_names = pipe_config.model_outputs  # else None
            if self.stage > 0:
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage -
                                                                 1]
            else:
                self.ranks_in_previous_stage = []

            if self.stage < self.num_stages - 1:
                self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
            else:
                self.ranks_in_next_stage = []
        else:
            self.target_tensor_names = None
            self.ranks_in_previous_stage = None
            self.ranks_in_next_stage = None

        self.training_tensor_shapes = self.get_shapes(bs_train)
        self.eval_tensor_shapes = self.get_shapes(bs_eval)

        self.training_tensor_dtypes = self.eval_tensor_dtypes = pipe_config.all_dtypes(
        )

        # Grad requirements for input tensors
        self.req_grad = pipe_config.stages[self.stage].req_grad

        # Grad requirements for input tensors (infer)
        outputs_req_grad = dict()
        my_outputs = pipe_config.stages[self.stage].outputs
        for i, stage in pipe_config.stages.items():
            for name, r in stage.req_grad.items():
                if name in my_outputs:
                    outputs_req_grad[name] = r

        self.outputs_req_grad = outputs_req_grad

    def comm_init_args(self):
        return (self.receive_ranks, self.send_ranks, self.tensor_tags,
                self.target_tensor_names, self.ranks_in_previous_stage,
                self.ranks_in_next_stage, self.TOTAL_TAGS, self.req_grad,
                self.outputs_req_grad, self.pipe_config)

    def get_shapes(self, batch_size):
        pipe_config = self.pipe_config
        pipe_config.change_batch(batch_size, for_replicated=True)
        return pipe_config.shapes()


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
