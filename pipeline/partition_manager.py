import torch
import logging
from typing import Dict
from collections import OrderedDict
from . import CommunicationHandlerBase
from .partition import Partition, LastPartition, FirstPartition
from .training.interface import PartitionedTrainer
from .tasks import DLTask
from .weight_prediction.interface import WeightPredictor
from .gap_aware import GapAware  # TODO: change to interface.
from .work_schedulers import WorkScheduler
from .weight_stashing import WeightStasher

# from gpu_mem_track import MemTracker
# import time


class SinglePartitionManager:
    def __init__(self, stage, configs: Dict, partition: torch.nn.Module, comm_handler: CommunicationHandlerBase,
                 work_scheduler: WorkScheduler,
                 training_tensor_shapes, eval_tensor_shapes,
                 device, is_last_partition, is_first_partition, statistics=None):

        if is_last_partition:
            partition_cls = LastPartition
        elif is_first_partition:
            partition_cls = FirstPartition
        else:
            partition_cls = Partition

        self.partition = partition_cls(partition, device, to_device=True)
        self.comm_handler = comm_handler
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition
        self.stage = stage
        self.num_stages = len(configs)
        self.work_scheduler = work_scheduler()

        self.weight_predictor = None
        self.gap_aware = None
        self.weight_stasher = None

        # State for train logging
        self.log_frequency = 100  # FIXME: magic number
        self.batches = 0

        self.fwd_rcev_buffers = None
        self.bwd_rcev_buffers = None

        # Async handle objects
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

        self.logger = logging.getLogger("msnag")
        # self.dl_iter = None

        # self.mb_to_fix_by_ga = {}
        self.modify_gradients_before_send = True  # FIXME add as option

        # Hints
        self.task: DLTask
        self.trainer: PartitionedTrainer
        self.weight_predictor: WeightPredictor
        self.gap_aware: GapAware
        self.weight_stasher: WeightStasher

    def set_task(self, task: DLTask):
        self.task = task

    def set_trainer(self, trainer: PartitionedTrainer):
        self.trainer = trainer

    def set_dataloader(self, dataloader):
        assert self.is_first_partition
        self.dataloader = dataloader
        self.dl_iter = iter(self.dataloader)

    def set_weight_predictor(self, weight_predictor: WeightPredictor, nag_with_predictor: bool):
        self.weight_predictor = weight_predictor
        self.nag_with_predictor = nag_with_predictor

    def set_gap_aware(self, gap_aware):
        self.gap_aware = gap_aware

    def set_weight_stasher(self, weight_stasher: WeightStasher):
        if self.is_last_partition and not (weight_stasher is None) :
            raise NotImplementedError()

        self.weight_stasher = weight_stasher

    def train(self):
        self.comm_handler.set_tensor_shapes(self.training_tensor_shapes)

        self.partition.train()

        # Handles the transition : eval -> train
        if not (self.fwd_rcev_buffers is None):
            self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                self.device)

    def eval(self):
        self.comm_handler.set_tensor_shapes(self.eval_tensor_shapes)

        self.partition.eval()

        # Handles the transition : train -> eval
        if not (self.fwd_rcev_buffers is None):
            self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                self.device)

    def run_batch_forward(self, batch_idx, num_batches, done_bwds=None):
        if self.is_first_partition:
            data = next(self.dl_iter)
            # TODO: handle y with separate coordinated dataloader
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

        else:
            if not self.fwd_rcev_buffers:
                self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                    self.device)
            x = self.fwd_rcev_buffers

            request_objects = self.comm_handler.recv_activations(x, batch_idx)

            # recv for fwd
            for obj in request_objects:
                # print(f"-I- {self.stage} waiting on rcv")
                while(not obj.is_completed()):
                    pass
                    # obj.wait()
                # print(f"-I- {self.stage} DONE waiting on rcv")

            # For comm handler with chunks we have to fix. For others its no-op.
            x = self.comm_handler.fix_after_recv(x)

            x, *ctx = self.task.unpack_data_for_partition(x)

            # TODO: last partition can do bengio nesterov instead of predicting.
            # Requires per partition optimizer config, or some hack.
            if self.weight_predictor and self.partition.training:
                self.weight_predictor.setup(
                    self.expected_staleness(batch_idx, done_bwds))
                # self.mb_to_fix_by_ga[batch_idx] = ((batch_idx - done_bwds) == 1)
                self.weight_predictor.forward()

                # Stash parameters for later.
                # Note: wait stasher should not be in last partition.
                # and not self.is_last_partition
                if self.weight_stasher and self.partition.training:
                    self.weight_stasher.stash_current(batch_idx)

                x = self.partition(x, batch_idx)
                self.weight_predictor.revert()
            else:
                # Stash parameters for later.
                if self.weight_stasher and self.partition.training:
                    self.weight_stasher.stash_current(batch_idx)

                x = self.partition(x, batch_idx)

            if not self.is_last_partition:
                send_ctx = self.task.pack_send_context(x, *ctx)
                request_objects = self.comm_handler.send_activations(
                    send_ctx, batch_idx)
            else:
                # Last partition
                if not self.partition.training:
                    # In Eval: Just calculate statistics.
                    self.trainer.calc_test_stats(x, *ctx)
                    return []

                # Backprop
                step_and_stats_ctx = self.trainer.backprop_last_partition(
                    x, *ctx)

                # Send partition border gradients
                grads = self.partition.get_grad(batch_idx)
                if isinstance(grads, torch.Tensor):
                    grads = (grads,)
                request_objects = self.comm_handler.send_gradients(
                    grads, batch_idx)

                # Step
                self.trainer.last_partition_step_and_statistics(
                    x, *ctx, step_and_stats_ctx)
                del x, ctx, step_and_stats_ctx

                # Print training statistics.
                if self.partition.training:
                    self.batches += 1
                    if self.batches % self.log_frequency == 0:
                        batch_log_str = ''
                        if hasattr(self.trainer, "scheduler"):
                            # Note: could be more than one LR, but we ignor this for simplicity.
                            lr = self.trainer.scheduler.get_last_lr()[0]
                            batch_log_str += '| lr {:02.4f}'.format(lr)

                        # TODO: add more stats. e.g can print here time, ' ms/batch {:5.2f} | ' ,...
                        self.logger.info(batch_log_str)

        return request_objects

    def run_batch_backward(self, batch_idx):
        # TODO: implement

        if not self.bwd_rcev_buffers:
            self.bwd_rcev_buffers = self.comm_handler.create_gradients_rcv_buffers(
                self.device)
        g = self.bwd_rcev_buffers

        # Solution to the DAMN bug with 4 partitions.
        # TODO: understnad why zero_() is the solution
        # I added detach just in case.
        for b in g:
            # b.detach_()
            b.detach_().zero_()
            # b.zero_()
            # if not (b.grad is None):
            #     b.grad._zero()

        request_objects = self.comm_handler.recv_gradients(g, batch_idx)

        # recv for bwd
        for obj in request_objects:
            while not obj.is_completed():
                pass
                # obj.wait()

        # TODO: maybe a different fix for gradients?
        g = self.comm_handler.fix_after_recv(g)

        if self.weight_stasher:
            # FIXME
            raise NotImplementedError()
            
            # Possible problem:
            # what we are about to do is:
            # restore stashed wieghts.data
            # compute backward on them
            # restore back current, up to date weights.data, with restore_last().
            # however, there is possibility that current `batch_index` is not last.
            # (This can happen in case of several backwards one after another)
            # In this case, we need to stash the currect version of weight so it will be the last.
            # This is problematic,
            # TODO
            self.weight_stasher.stash_if_current_is_last(batch_idx)
            # Restore to parameters which the fwd was run on
            self.weight_stasher.pop_restore_stashed(batch_idx)

        # Compute gradeints
        self.partition.recompute_and_backward(g, batch_idx)

        # TODO: we can modify just the gradeint of the first layer.
        if self.modify_gradients_before_send:
            # Modify gradients
            self.trainer.modify_gradients()

        if self.weight_stasher:
            # Restore to previosly saved parameters, we we can do the step on them.
            self.weight_stasher.restore_last()
            # TODO: look to pipedream implementation and udnerstand what they do with the weight decay.

        # Step and statistics
        request_objects = None
        # we may want to get the grad for sending before the step.
        if not (self.is_first_partition):
            g = self.partition.get_grad(batch_idx)
            # TODO: this is super ugly. Caused huge bug.
            if isinstance(g, torch.Tensor):
                g = (g,)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)

        if not self.modify_gradients_before_send:
            # Modify gradients
            self.trainer.modify_gradients()

        self.trainer.non_last_partition_step()
        return request_objects

    def expected_staleness(self, done_fwds, done_bwds):
        if self.nag_with_predictor and ((done_fwds - done_bwds) == 0):
            return 1
        else:
            return done_fwds - done_bwds

    def run_forward_until_flush(self, num_batches):

        for done_fwds in range(num_batches):

            sent_request_objects = self.run_batch_forward(done_fwds, num_batches)
            if sent_request_objects:  # last partition returns empty list.
                self.async_fwd_objects[done_fwds] = sent_request_objects

            if len(self.async_fwd_objects) > 1:
                # Pop the item that was increaced first.
                _, (tmp_send_objects, tmp_sent_items) = self.async_fwd_objects.popitem(
                    last=False)
                for i in tmp_send_objects:
                    while not i.is_completed():
                        pass
                        # i.wait()

        # Also clear in the end, just in case...
        for (sent_request_objects, tmp_sent_items) in self.async_fwd_objects.values():
            for i in sent_request_objects:
                while not i.is_completed():
                    pass
                    # i.wait()

        self.async_fwd_objects.clear()
        # FIXME: not sure if this needed.
        # For now I leave this for debugging/safety.

        if not self.comm_handler.cpu:
            # HACK: synchronize.
            torch.cuda.synchronize(device=self.device)

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called (if first partition)
            train() or eval() was called
        """
        done_bwds = 0
        done_fwds = 0

        num_steps = num_batches
        while done_bwds < num_batches:
            # for step_index in range(num_steps):
            # Act according to some policy
            action_is_fwd = self.work_scheduler(self.stage, self.num_stages,
                                                num_steps, done_fwds, done_bwds)
            if action_is_fwd:
                sent_request_objects = self.run_batch_forward(
                    done_fwds, num_batches, done_bwds)
                # FIXME: last partition inserts its gradints into async_fwd_objects,
                # it works, but it can be trouble.
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
            # TODO: can write the entire thing MUCH more nicely
            # if we just save asside and insert the new objects at the end.

            # wait on the first,
            if len(self.async_fwd_objects) > 1:
                # Pop the item that was increaced first.
                _, (sent_request_objects, tmp_sent_items) = self.async_fwd_objects.popitem(
                    last=False)
                for i in sent_request_objects:
                    while(not i.is_completed()):
                        pass
                        # i.wait()

            if len(self.async_bwd_objects) > 1:
                # Pop the item that was increaced first.
                _, (sent_request_objects, tmp_sent_items) = self.async_bwd_objects.popitem(
                    last=False)
                for i in sent_request_objects:
                    while(not i.is_completed()):
                        pass
                        # i.wait()

        # FIXME: remove this print later, its used to debug how much we have in pipe.
        print(self.stage, len(self.async_bwd_objects),
              len(self.async_fwd_objects))
        # FIXME: maybe more than 1.
        while len(self.async_fwd_objects) > 0:
            _, (o1, t1) = self.async_fwd_objects.popitem(last=False)
            for i in o1:
                while(not i.is_completed()):
                    pass
                    # i.wait()

        while len(self.async_bwd_objects) > 0:
            _, (o2, t2) = self.async_bwd_objects.popitem(last=False)
            for i in o2:
                while(not i.is_completed()):
                    pass
                    # i.wait()

        if not self.comm_handler.cpu:
            # HACK: synchronize.
            torch.cuda.synchronize(device=self.device)
