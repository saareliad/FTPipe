import time
import torch
import logging
from typing import Dict

from . import CommunicationHandler
from .partition import Partition, LastPartition, FirstPartition
from .training.interface import AnyTrainer


class SinglePartitionManager:
    # FIXME: to the partitionion class we use...
    def __init__(self, stage, configs: Dict, partition: torch.nn.Module, comm_handler: CommunicationHandler,
                 training_tensor_shapes, eval_tensor_shapes,
                 device, is_last_partition, is_first_partition):

        # self.split_dim = split_dim
        # self.input_names = configs.pop('model inputs')
        # self.output_names = configs.pop('model outputs')

        if is_last_partition:
            partition_cls = LastPartition
        elif is_first_partition:
            partition_cls = FirstPartition
        else:
            partition_cls: Partition

        self.partition = partition_cls(partition, device, to_device=True)
        self.comm_handler = comm_handler
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shape = eval_tensor_shapes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition
        self.stage = stage
        self.num_stages = len(configs)  # FIXME:

        self.fwd_rcev_buffers = None
        self.bwd_rcev_buffers = None

        self.trainer = None

        # Async handle objects
        self.async_fwd_objects = {}
        self.async_bwd_objects = {}

        self.logger = logging.getLogger("msnag")

    def set_trainer(self, trainer: AnyTrainer):
        self.trainer = trainer

    def set_dataloader(self, dataloader):
        if self.is_first_partition:
            self.dataloader = dataloader

    def train(self):
        # self.tensors = []
        # self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        # self.forward_minibatch_id = 0
        # self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)

        self.partition.train()

    def eval(self):
        # self.tensors = []
        # self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        # self.tensor_shapes["ack"] = (1,)
        self.forward_only = True  # TODO: work that out ....

        # self.forward_minibatch_id = 0
        # self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)

        self.partition.eval()

    def run_batch_forward(self, batch_idx):
        if self.is_first_partition:
            data = next(self.dl_iter)
            # TODO: handle y with separate coordinated dataloader
            # TODO: generic data handling.
            # Can be according to trainer.
            assert len(data) == 2
            x, y = data
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            x = self.partition(x, batch_idx)
            request_objects = self.comm_handler.send_activations(
                (*x, y), batch_idx)
        else:
            if not self.fwd_rcev_buffers:
                self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                    self.device)
            x = self.fwd_rcev_buffers

            request_objects = self.comm_handler.recv_activations(x, batch_idx)

            # recv for fwd
            for obj in request_objects:
                obj.wait()

            x, y = x[:-1], x[-1]
            x = self.partition(x, batch_idx)

            if not self.is_last_partition:
                request_objects = self.comm_handler.send_activations(
                    (*x, y), batch_idx)
            else:
                # Last partition
                # Also do backward and step
                self.trainer.do_your_job(x, y)
                # TODO: save the gradient - (later to be passed to previous partition)
                # mb_to_last_res[micro_batch] = partition.get_grad(micro_batch)
                grads = self.partition.get_grad(batch_idx)
                # Send gradients async
                request_objects = self.comm_handler.send_gradients(
                    grads, batch_idx)

        return request_objects

    def policy_scheduler_is_fwd(self, num_steps, done_bwds, done_fwds):
        # TODO: implement...
        # and generically/nicely.
        # FIXME: Here is a dummy sequential implementation with dummy args.
        # stage = self.stage
        # num_stages = self.num_stages

        POLICY = 'P1'

        if POLICY == 'SEQ':
            if self.is_last_partition:
                return True
            else:
                if not (done_bwds < done_fwds):
                    return True
                else:
                    return False
        elif POLICY == 'P1':
            if self.is_last_partition:
                return True
            if done_fwds == num_steps:
                return False
            allowed_staleness = self.num_stages - 1 - self.stage
            current_staleness = done_fwds - done_bwds
            return current_staleness <= allowed_staleness

        # return True

    def run_batch_backward(self, batch_idx):
        # TODO: implement

        if not self.bwd_rcev_buffers:
            self.bwd_rcev_buffers = self.comm_handler.create_gradients_rcv_buffers(
                self.device)
        g = self.bwd_rcev_buffers

        request_objects = self.comm_handler.recv_gradients(g, batch_idx)

        # recv for bwd
        for obj in request_objects:
            obj.wait()

        self.partition.recompute_and_backward(g, batch_idx)
        self.trainer.step_on_computed_grads()

        if not (self.is_first_partition):
            g = self.Partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)
            return request_objects

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called
            train() or eval()
        """
        done_bwds = 0
        done_fwds = 0

        if self.is_first_partition:
            self.dl_iter = iter(self.dataloader)

        # if not (self.is_last_partition) else num_batches // 2
        num_steps = num_batches
        while done_bwds < num_batches:
            # for step_index in range(num_steps):
            # Act according to some policy
            action_is_fwd = self.policy_scheduler_is_fwd(
                num_steps, done_bwds, done_fwds)
            if action_is_fwd:
                sent_request_objects = self.run_batch_forward(done_fwds)
                self.async_fwd_objects[done_fwds] = sent_request_objects
            else:
                sent_request_objects = self.run_batch_backward(done_bwds)
                if not (self.is_first_partition):
                    self.async_bwd_objects[done_bwds] = sent_request_objects
            # Increase counters
            if self.is_last_partition:
                done_bwds += 1
                done_fwds += 1
            else:
                if action_is_fwd:
                    done_fwds += 1
                else:
                    done_bwds += 1

            # FIXME: we may not want to wait on this yet.
            # TODO: save the object and wait only when truly needed.

        # TODO: wait on all objects
        for sent_request_objects in self.async_bwd_objects.values():
            for i in sent_request_objects:
                i.wait()

        for sent_request_objects in self.async_fwd_objects.values():
            for i in sent_request_objects:
                i.wait()

        self.async_bwd_objects.clear()
        self.async_fwd_objects.clear()

        self.logger.info(f"Done running until flush stage:{self.stage}")
        torch.distributed.barrier()
