import time
import torch
import logging
from typing import Dict
from collections import OrderedDict
from . import CommunicationHandler
from .partition import Partition, LastPartition, FirstPartition
from .training.interface import AnyTrainer
from .tasks import DLTask
from .weight_prediction.interface import WeightPredictor
from .itertools import grouper

# from gpu_mem_track import MemTracker


class SinglePartitionManager:
    def __init__(self, stage, configs: Dict, partition: torch.nn.Module, comm_handler: CommunicationHandler,
                 training_tensor_shapes, eval_tensor_shapes,
                 device, is_last_partition, is_first_partition, statistics=None):

        if is_last_partition:
            partition_cls = LastPartition
        elif is_first_partition:
            partition_cls = FirstPartition
        else:
            partition_cls: Partition

        self.partition = partition_cls(partition, device, to_device=True)
        self.comm_handler = comm_handler
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition
        self.stage = stage
        self.num_stages = len(configs)  # FIXME:
        self.weight_predictor = None

        # State for train logging
        self.log_frequency = 100  # FIXME: magic number
        self.batches = 0

        self.fwd_rcev_buffers = None
        self.bwd_rcev_buffers = None

        # Async handle objects
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

        # # TODO: read real values from config...
        # self.base_fwd_send_tensors = 1
        # self.base_bwd_send_tensors = 1

        self.logger = logging.getLogger("msnag")
        self.dl_iter = None

        # Hints
        self.task: DLTask
        self.trainer: AnyTrainer
        self.weight_predictor: WeightPredictor

    def set_task(self, task: DLTask):
        self.task = task

    def set_trainer(self, trainer: AnyTrainer):
        self.trainer = trainer

    def set_dataloader(self, dataloader):
        if self.is_first_partition:
            self.dataloader = dataloader
            self.dl_iter = iter(self.dataloader)

    def set_weight_predictor(self, weight_predictor: WeightPredictor, nag_with_predictor: bool):
        self.weight_predictor = weight_predictor
        self.nag_with_predictor = nag_with_predictor

    def train(self):
        self.tensor_shapes = self.training_tensor_shapes
        # self.forward_only = False

        self.comm_handler.set_tensor_shapes(self.tensor_shapes)

        self.partition.train()

        # Handles the transition : eval -> train
        if not (self.fwd_rcev_buffers is None):
            self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                self.device)

    def eval(self):
        self.tensor_shapes = self.eval_tensor_shapes

        # self.tensor_shapes["ack"] = (1,)
        # self.forward_only = True  # TODO: work that out.

        self.comm_handler.set_tensor_shapes(self.tensor_shapes)

        self.partition.eval()

        # Handles the transition : train -> eval
        if not (self.fwd_rcev_buffers is None):
            self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                self.device)

    def run_batch_forward(self, batch_idx, done_bwds=None):
        if self.is_first_partition:
            data = next(self.dl_iter)
            # TODO: handle y with separate coordinated dataloader
            # TODO: generic data handling.
            # Can be according to trainer.

            x, *ctx = self.task.unpack_data_for_partition(data)

            if self.weight_predictor and self.partition.training:
                self.weight_predictor.setup(
                    self.expected_staleness(batch_idx, done_bwds))
                self.weight_predictor.forward()
                x = self.partition(x, batch_idx)
                self.weight_predictor.revert()
            else:
                x = self.partition(x, batch_idx)

            send_ctx = self.task.pack_send_context(x, *ctx)
            # print("Sending", *ctx, ctx[0].shape)
            # for i in send_ctx:
            #     print(f"send ctx: {i.shape}")

            request_objects = self.comm_handler.send_activations(
                send_ctx, batch_idx)

            # # FIXME
            # # FIXME
            # # FIXME
            # for i in request_objects:
            #     i.wait()

        else:
            if not self.fwd_rcev_buffers:
                self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                    self.device)
            x = self.fwd_rcev_buffers

            # x = self.comm_handler.create_activations_recv_buffers(self.device)

            request_objects = self.comm_handler.recv_activations(x, batch_idx)
            
            # recv for fwd
            for obj in request_objects:
                # print(f"-I- {self.stage} waiting on rcv")
                obj.wait()
                # print(f"-I- {self.stage} DONE waiting on rcv")
            
            x = [torch.cat(group) for group in grouper(x, self.comm_handler.num_chunks)]

            x, *ctx = self.task.unpack_data_for_partition(x)

            # TODO: last partition can do bengio nesterov instead of predicting.
            if self.weight_predictor and self.partition.training:
                self.weight_predictor.setup(
                    self.expected_staleness(batch_idx, done_bwds))
                self.weight_predictor.forward()
                x = self.partition(x, batch_idx)
                self.weight_predictor.revert()
            else:
                x = self.partition(x, batch_idx)

            if (not self.partition.training) and self.is_last_partition:
                # print(*ctx, ctx[0].shape)
                self.trainer.calc_test_stats(x, *ctx)
                return []

            # if self.partition.training:
            if not self.is_last_partition:
                send_ctx = self.task.pack_send_context(x, *ctx)
                request_objects = self.comm_handler.send_activations(
                    send_ctx, batch_idx)
            else:
                # Last partition
                # Also do backward and step
                self.trainer.do_your_job(x, *ctx)
                # TODO: save the gradient - (later to be passed to previous partition)
                grads = self.partition.get_grad(batch_idx)
                # TODO: this is a little ugly, maybe find a better way later
                if isinstance(grads, torch.Tensor):
                    grads = (grads,)
                # Send gradients async
                request_objects = self.comm_handler.send_gradients(
                    grads, batch_idx)

                # TODO: can print here
        #          {:5d}/{:5d} batches | '
        # #           'lr {:02.2f} | ms/batch {:5.2f} | '
        # #           'loss {:5.2f} | ppl {:8.2f}'

                # if self.is_last_partition
                if self.partition.training:
                    self.batches += 1
                    if self.batches % self.log_frequency == 0:
                        # TODO: scheduler could be None.
                        # TODO: could be more than one LR
                        lr = self.trainer.scheduler.get_last_lr()[0]
                        # TODO: add more stats..
                        log_str = '| lr {:02.4f}'.format(lr)
                        self.logger.info(log_str)

                for i in self.fwd_rcev_buffers:
                    i.grad = None

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

        g = [torch.cat(group) for group in grouper(g, self.comm_handler.num_chunks)]
        # g = torch.cat(g, self.comm_handler.

        self.partition.recompute_and_backward(g, batch_idx)
        self.trainer.step_on_computed_grads()

        # for z in self.bwd_rcev_buffers:
        #     z.detach_().zero_()

        if not (self.is_first_partition):
            g = self.partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)

            return request_objects

    def expected_staleness(self, done_fwds, done_bwds):
        if self.nag_with_predictor:
            return min(1, done_fwds - done_bwds)
        else:
            return done_fwds - done_bwds

    def run_forward_until_flush(self, num_batches):
        # TODO: write fwd only for last partition..
        # if self.is_first_partition and (self.dl_iter is None):
        #     self.dl_iter = iter(self.dataloader)

        for done_fwds in range(num_batches):

            sent_request_objects = self.run_batch_forward(done_fwds)

            self.async_fwd_objects[done_fwds] = sent_request_objects

            if len(self.async_fwd_objects) > 1:
                # Pop the item that was increaced first.
                _, tmp_send_objects = self.async_fwd_objects.popitem(
                    last=False)
                for i in tmp_send_objects:
                    i.wait()

            # TODO, wait on first

            # TODO: don't wait every time, add option to accum by depth

        # Also clear in the end, just in case...
        for sent_request_objects in self.async_fwd_objects.values():
            for i in sent_request_objects:
                i.wait()

        self.async_fwd_objects.clear()
        # FIXME: not sure if this needed.
        # For now I leave this for debugging/safety.

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called
            train() or eval()
        """
        done_bwds = 0
        done_fwds = 0

        # if self.is_first_partition and (self.dl_iter is None):
        #     self.dl_iter = iter(self.dataloader)

        # if not (self.is_last_partition) else num_batches // 2
        num_steps = num_batches
        while done_bwds < num_batches:
            # for step_index in range(num_steps):
            # Act according to some policy
            action_is_fwd = self.policy_scheduler_is_fwd(
                num_steps, done_bwds, done_fwds)
            if action_is_fwd:
                sent_request_objects = self.run_batch_forward(
                    done_fwds, done_bwds)
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
            # TODO: can write the entire thing MUCH more nicely if we just save asside and insert the new objects at the end...

            # wait on the first,
            if len(self.async_fwd_objects) > 1:
                # Pop the item that was increaced first.
                _, sent_request_objects = self.async_fwd_objects.popitem(
                    last=False)
                for i in sent_request_objects:
                    i.wait()

            if len(self.async_bwd_objects) > 1:
                # Pop the item that was increaced first.
                _, sent_request_objects = self.async_bwd_objects.popitem(
                    last=False)
                for i in sent_request_objects:
                    i.wait()

        # FIXME: just for even numbers.
        print(self.stage, len(self.async_bwd_objects), len(self.async_fwd_objects))
        # assert len(self.async_bwd_objects) == 1
        # assert len(self.async_fwd_objects) == 1

        # FIXME: maybe more than 1...
        while len(self.async_fwd_objects) > 0:
            _, o1 = self.async_fwd_objects.popitem(last=False)                    
            for i in o1:
                i.wait()
    
        while len(self.async_bwd_objects) > 0:
            _, o2 = self.async_bwd_objects.popitem(last=False)
            for i in o2:
                i.wait()

        # # TODO: wait on all objects
        # for sent_request_objects in self.async_bwd_objects.values():
        #     for i in sent_request_objects:
        #         i.wait()

        # for sent_request_objects in self.async_fwd_objects.values():
        #     for i in sent_request_objects:
        #         i.wait()

        # self.async_bwd_objects.clear()
        # self.async_fwd_objects.clear()

        # self.logger.info(f"Done running until flush stage:{self.stage}")
        # torch.distributed.barrier()
