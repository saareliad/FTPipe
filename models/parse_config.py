from .models import AVAILABLE_MODELS

from itertools import count, chain
from collections import OrderedDict, defaultdict
from .simple_partitioning_config import PipelineConfig
from pprint import pprint
from copy import deepcopy


def get_my_send_recv_ranks(pipe_config: PipelineConfig, stage_id, stage_to_rank_map=None, prefer_seq_sends=True):
    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    stages = pipe_config.d['stages']
    receive_ranks = OrderedDict()
    send_ranks = defaultdict(list)

    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            # Update only for this stage...
            if i != stage_id and j != stage_id:
                continue
            stage_i = stages[i]
            stage_j = stages[j]
            for tensor_name in stage_i['outputs']:
                if tensor_name in stage_j['inputs']:
                    if stage_id == j:  # recv
                        if tensor_name in receive_ranks:
                            if prefer_seq_sends:
                                print(f"-V- stage {stage_id}: preferring to recv from a later: {i}-->{j}: {tensor_name}")
                            else:
                                raise ValueError(f"Input {tensor_name} received from multiple stages")
                        receive_ranks[tensor_name] = ranks_in_stage(i)
                    else:  # stage == i, send
                        if prefer_seq_sends:
                            # check if I'm the closest
                            all_sending_dist = [x for x, v in stages.items() if tensor_name in v['outputs'] and x < j]
                            assert len(all_sending_dist) > 0
                            assert stage_id in all_sending_dist
                            closest_sender = max(all_sending_dist)
                            if stage_id != closest_sender:
                                print(f"-v- stage {stage_id}: will not send {i}-->{j}: {tensor_name}."
                                      f" There is a closer sender: {closest_sender}")
                                continue
                        send_ranks[tensor_name].extend(ranks_in_stage(j))

    # Ensure the sort order is like config.
    send_ranks = OrderedDict(
        (k, send_ranks[k]) for k in stages[stage_id]['outputs'] if k in send_ranks)
    receive_ranks = OrderedDict((k, receive_ranks[k])
                                for k in stages[stage_id]['inputs']
                                if k in receive_ranks)

    return send_ranks, receive_ranks


def get_stage_to_rank_map(pipe_config: PipelineConfig, stage_to_device_map=None, stateless_tied_same_process=False):
    counter = count()
    # Create stage_to_rank_map
    # (1) normal
    # (2) tied wieghts, using the same process (And multithreading)
    if not stateless_tied_same_process:
        stage_to_rank_map = {
            i: [next(counter) for _ in stage['devices']]
            for i, stage in pipe_config.d['stages'].items()
        }
    else:
        assert stage_to_device_map is not None
        assert len(stage_to_device_map) == len(pipe_config.d['stages'])
        stage_to_rank_map = {}
        record_ranks = {}
        for i, stage in pipe_config.d['stages'].items():
            tied_to = [
                k for k, j in enumerate(stage_to_device_map)
                if j == stage_to_device_map[i]
            ]
            if len(tied_to) > 1 and i == min(tied_to):
                my_ranks = [next(counter) for _ in stage['devices']]
                for k in tied_to:
                    record_ranks[i] = my_ranks
            elif len(tied_to) > 1 and i != min(tied_to):
                my_ranks = record_ranks[i]
            else:
                my_ranks = record_ranks[i]

            stage_to_rank_map[i] = my_ranks
    return stage_to_rank_map


class PartitioningConfigParser:
    def __init__(self,
                 cfg,
                 rank,
                 bs_train,
                 bs_eval,
                 handler=None,
                 send_target_in_pipe=False,
                 stage_to_device_map=None,
                 stateless_tied_same_process=False,
                 prefer_seq_sends=True):

        if handler is None:
            handler = AVAILABLE_MODELS.get(cfg)
            if handler is None:
                raise ValueError(f"Model {cfg} not found. AVAILABLE_MODELS={AVAILABLE_MODELS.keys()}")
        pipe_config = handler.get_pipe_config()
        model = handler.realize_stage_for_rank(batch_size=bs_train, my_rank=rank)

        self.stage = pipe_config.rank_to_stage_idx(rank)
        self.pipe_config = pipe_config

        self.model = model
        self.num_stages = pipe_config.n_stages

        stage_to_rank_map = get_stage_to_rank_map(pipe_config, stage_to_device_map=stage_to_device_map,
                                                  stateless_tied_same_process=stateless_tied_same_process)

        self.send_ranks, self.receive_ranks = get_my_send_recv_ranks(pipe_config, self.stage,
                                                                     stage_to_rank_map=stage_to_rank_map,
                                                                     prefer_seq_sends=prefer_seq_sends)

        from pprint import pprint
        pprint(f"Stage: {self.stage} send_ranks: {self.send_ranks}")
        pprint(f"Stage: {self.stage} receive_ranks: {self.receive_ranks}")

        # Handle sending target in pipe. (deprecated)
        if send_target_in_pipe:
            self.target_tensor_names = pipe_config.d['model_outputs']  # else None
            if self.stage > 0:
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
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

        self.eval_tensor_shapes = self.get_shapes(bs_eval)
        self.training_tensor_shapes = self.get_shapes(bs_train)

        self.training_tensor_dtypes = self.eval_tensor_dtypes = pipe_config.get_dtypes_for_stage(self.stage)

        # Grad requirements for input tensors
        # FIXME: the could be done smarter
        self.req_grad = pipe_config.get_inputs_req_grad_for_stage(self.stage)

        # Grad requirements for output tensors (infer)
        # FIXME:
        self.outputs_req_grad = pipe_config.get_outputs_req_grad_for_stage(self.stage)

        _check_shared_parameters(pipe_config)

    def comm_init_args(self):
        return (self.receive_ranks, self.send_ranks,
                self.target_tensor_names, self.ranks_in_previous_stage,
                self.ranks_in_next_stage, self.req_grad,
                self.outputs_req_grad, self.pipe_config)

    def get_shapes(self, batch_size):
        pipe_config = self.pipe_config
        pipe_config.change_batch(batch_size, for_replicated=True)
        return deepcopy(pipe_config.get_shapes_for_stage(self.stage))


# model
# num_stages
# stage

# Comm handler:

# receive_ranks,
# send_ranks,
# target_tensor_names,
# ranks_in_previous_stage,
# ranks_in_next_stage,

# Partition Manager:

# training_tensor_dtypes,
# training_tensor_shapes,
# eval_tensor_shapes
def is_shared_parameter(tensor_scope):
    # HACK. (can do it also at config)
    return "Parameter" in tensor_scope


def _check_shared_parameters(pipe_config: PipelineConfig):
    shared = defaultdict(set)
    for i, s in pipe_config.d['stages'].items():
        for n in chain(s['inputs'], s['outputs']):
            if is_shared_parameter(n):
                shared[i].add(n)

    if shared:
        pprint(f"Shared Parameters: {shared}")

    return shared
