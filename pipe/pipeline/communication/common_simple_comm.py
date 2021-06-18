import logging
import warnings
from abc import ABC
from collections import OrderedDict

import torch
import torch.distributed as dist

from pipe.models.simple_partitioning_config import PipelineConfig
from .interface import CommunicationHandlerBase
from .tags import tensor_tags_from_config


class SimpleCommBase(CommunicationHandlerBase, ABC):
    """ common for all MPI based.
    TODO: some functions in the end should be moved to lower class
    """

    def __init__(
            self,
            rank,
            local_rank,
            backend,
            world_size,
            num_stages,
            stage,
            receive_ranks,
            send_ranks,
            target_tensor_names,
            ranks_in_previous_stage,  # NOTE: deprecated
            ranks_in_next_stage,  # NOTE: deprecated
            req_grad,
            outputs_req_grad,
            pipe_config: PipelineConfig,
            cpu,
            num_chunks,
            device,
            GRAD_UGLY_SHAMEFUL_NAME="_grad",
            verbose=False):
        super().__init__()

        # NOTE: Order is important, must call for send,recv ranks before in/out req grads.
        self.tensor_dtypes = None

        # inputs/outputs are not part of send/recv ranks
        for to_del in [receive_ranks, send_ranks]:
            for inout in [pipe_config.d['model_inputs'], pipe_config.d['model_outputs']]:
                for i in inout:
                    if i in to_del:
                        del to_del[i]

        assert isinstance(GRAD_UGLY_SHAMEFUL_NAME, str)
        self.GRAD_UGLY_SHAMEFUL_NAME = GRAD_UGLY_SHAMEFUL_NAME
        self.verbose = verbose
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.logger = logging.getLogger('msnag')
        self.stage = stage
        self.pipe_config = pipe_config

        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks

        # Do not calculate and send gradients for tensors which do not req grad.
        self.tensors_names_with_no_grad = set()
        for i, v in req_grad.items():
            assert isinstance(v, bool)
            if isinstance(v, bool):
                if not v:
                    self.tensors_names_with_no_grad.add(i)
        # Do not receive gradients for tensors which do not req grad.
        for i, v in outputs_req_grad.items():
            assert isinstance(v, bool), str((i, v))
            if not v:
                self.tensors_names_with_no_grad.add(i)

        # Optional
        if target_tensor_names:
            self.tensors_names_with_no_grad.update(target_tensor_names)

        self.cpu = cpu
        self.device = device
        self.world_size = world_size

        self.num_chunks = num_chunks  # optionally split the batches to chunks

        self.activations_rcv_items = list(self.receive_ranks.items())

        self.grad_rcv_items_without_extention = [
            (i, v) for i, v in self.send_ranks.items()
            if i not in self.tensors_names_with_no_grad
        ]

        self.grad_send_items_without_extention = [
            (i, v) for i, v in self.receive_ranks.items()
            if i not in self.tensors_names_with_no_grad
        ]

        self.grad_rcv_items = [(i + GRAD_UGLY_SHAMEFUL_NAME, v)
                               for i, v in self.send_ranks.items()
                               if i not in self.tensors_names_with_no_grad]

        self.grad_send_items = [(i + GRAD_UGLY_SHAMEFUL_NAME, v)
                                for i, v in self.receive_ranks.items()
                                if i not in self.tensors_names_with_no_grad]

        self.grad_rcv_dict_without_extention = OrderedDict(self.grad_rcv_items_without_extention)
        self.grad_send_dict_without_extention = OrderedDict(self.grad_send_items_without_extention)

        self.grad_send_dict = OrderedDict(self.grad_send_items)
        self.grad_rcv_dict = OrderedDict(self.grad_rcv_items)

        tag_info = tensor_tags_from_config(pipe_config)

        # if rank == 0:
        #     print("="*40)
        #     print("TAG INFO")
        #     print(tag_info)
        #     print("=" * 40)

        self.tensor_tags, self.TOTAL_TAGS = tag_info

        if target_tensor_names:
            self.ranks_in_next_stage = ranks_in_next_stage
            self._register_target_tensor(target_tensor_names,
                                         ranks_in_previous_stage,
                                         ranks_in_next_stage)  # If needed.

        # self.logger.debug(f"Send ranks: {self.send_ranks}")
        # self.logger.debug(f"Receive ranks: {self.receive_ranks}")

    def init_process_group(self, *args, **kw):

        backend = self.backend
        rank = self.rank
        local_rank = self.local_rank
        world_size = self.world_size

        # Initialize the distributed environment.
        dist.init_process_group(backend)
        assert dist.get_world_size() == world_size
        self.logger.info(
            f"Initialized process group; backend: {backend}, rank: {rank}, "
            f"local_rank: {local_rank}, world_size: {world_size}")

    def _register_target_tensor(self, target_tensor_names,
                                ranks_in_previous_stage, ranks_in_next_stage):
        warnings.warn("Sending targets in pipeline is deprecated.")
        #  Its inefficient to pass the targets all the way to the end, it is deprecated
        # It can be replaced by popper data loaders and timing.
        # However, when using dataloaders are in different machines,
        # we need to test and assert that the loading and shuffling is done in the same order.
        for target_tensor_name in target_tensor_names:
            if len(ranks_in_previous_stage) > 0:
                self.receive_ranks[
                    target_tensor_name] = ranks_in_previous_stage
            if len(self.ranks_in_next_stage) > 0:
                self.send_ranks[target_tensor_name] = ranks_in_next_stage

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def set_tensor_dtypes(self, tensor_dtypes):
        self.tensor_dtypes = tensor_dtypes

    def init_buffers_ctx(self, buffers_ctx):
        (
            training_tensor_shapes,
            eval_tensor_shapes,
            training_tensor_dtypes,
            eval_tensor_dtypes,
            last_batch_train_shapes,
            last_batch_test_shapes,
            max_buffers,
            keep_buffers_alive,
        ) = buffers_ctx

        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.eval_tensor_dtypes = eval_tensor_dtypes
        self.last_batch_train_shapes = last_batch_train_shapes
        self.last_batch_test_shapes = last_batch_test_shapes
        self.max_buffers = max_buffers
        self.keep_buffers_alive = keep_buffers_alive

        self.changed_shapes_last_batch_fwd = False
        self.changed_shapes_last_batch_bwd = False

        # print("last_batch_train_shapes", last_batch_train_shapes)

    def fix_after_recv(self, x, is_grad=False):
        """ Fixes received buffer after sync wait ends"""
        if is_grad:
            out = []
            ix = iter(x)
            for name, ranks in self.grad_rcv_items:
                if len(ranks) > 1:
                    tensors = [
                        t for t in [next(ix) for _ in range(len(ranks))]
                        if t is not None
                    ]
                    out.append(torch.stack(tensors).sum(0))
                else:
                    out.append(next(ix))
            return out
        return x
