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
from .work_schedulers import WorkScheduler, get_fwds_between_first_and_seconds_step_for_stage
from .weight_stashing import WeightStasher
from .buffer import Buffers
import numpy as np
import types

# from gpu_mem_track import MemTracker
# import time


class SinglePartitionManager:
    PROBLEMATIC_POLICY = 'SAME'
    # PROBLEMATIC_POLICY = 'SKIP'

    def __init__(self, stage, configs: Dict, partition: torch.nn.Module, comm_handler: CommunicationHandlerBase,
                 work_scheduler: WorkScheduler,
                 training_tensor_shapes, eval_tensor_shapes, training_tensor_dtypes,  # FIXME
                 device, is_last_partition, is_first_partition, log_frequency=100, max_buffers=2, step_every=1):

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
        self.training_tensor_dtypes = training_tensor_dtypes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition
        self.stage = stage
        self.num_stages = len(configs)

        self.step_every = step_every
        self.work_scheduler = work_scheduler(step_every)

        # HACK:
        # FIXME: num batches just have to be high enough for the calculation.
        fwds, is_problematic = get_fwds_between_first_and_seconds_step_for_stage(
            self.work_scheduler, self.stage, self.num_stages, num_batches=390)
        self.is_problematic = is_problematic
        if is_problematic:
            print(
                f"-V- Patching problematic batches {fwds} for stage {self.stage}")
            if self.PROBLEMATIC_POLICY == 'SKIP':
                self.set_problematic_skip()

        self.weight_predictor = None
        self.gap_aware = None
        self.weight_stasher = None

        # State for train logging
        self.log_frequency = log_frequency
        self.batches = 0

        # State for recv buffers
        if max_buffers > 2:  # FIXME
            raise NotImplementedError()
        self.fwd_rcev_buffers = Buffers(
            max_buffers, self.comm_handler.create_activations_recv_buffers,
            self.device, self.comm_handler.recv_activations, is_grad=False)
        self.bwd_rcev_buffers = Buffers(
            max_buffers, self.comm_handler.create_gradients_rcv_buffers,
            self.device, self.comm_handler.recv_gradients, is_grad=True)
        self.recv_all_bwd = True

        # Holds Async handle objects (for isends)
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

        self.logger = logging.getLogger("msnag")

        # self.modify_gradients_before_send = False  # TODO add as option
        self.delay_at_batch = {}

        # Hints,May be set later.
        # self.dl_iter = None
        self.task: DLTask
        self.trainer: PartitionedTrainer
        self.weight_predictor: WeightPredictor
        self.gap_aware: GapAware
        self.weight_stasher: WeightStasher

    def set_problematic_same(self):
        self.is_problematic = True
        pass

    def set_problematic_skip(self):
        """
        # The problem: Different versions in forward, same versions in backward due to ODD staleness.
        # 'SKIP' Changes the problem to same versions in forward, different version in backward
        # TODO: weight precition for the initial big batch
        """
        self.is_problematic = True
        se = self.step_every
        skip_batch_index = se - 1

        print(f"Stage {self.stage}. first batch:{list(range(se+1))}")

        def should_do_step_patch(self, batch_idx):
            # old_lrs = None
            if batch_idx <= skip_batch_index:
                # if batch_idx == skip_batch_index:
                #     print(f"-V- stage:{self.stage} Skipping problematic batch {skip_batch_index}")
                return False, None
            # elif batch_idx == small_step_batch:
            #     # Exceptional smaller than usual batch step
            #     factor = (se - 1) / se
            #     old_lrs, _ = self.scale_lr(factor)
            #     return True, old_lrs
            elif batch_idx == se:
                # TODO: Exceptional bigger than usual batch step
                factor = (se + 1) / se
                old_lrs, _ = self.scale_lr(factor)
                return True, old_lrs

            return (batch_idx % se) == 0,  None

        self.should_do_step = types.MethodType(should_do_step_patch, self)

        def get_micro_batch(self, batch_index):
            if batch_index <= se:
                return batch_index
            return (batch_index+1) % se

        self.get_micro_batch = types.MethodType(get_micro_batch, self)

    def get_micro_batch(self, batch_index):
        return batch_index % self.step_every

    def scale_lr(self, factor):
        # TODO:
        pgs = self.trainer.optimizer.param_groups

        old_lrs = []
        new_lrs = []
        for g in pgs:
            old_lr = g['lr']
            new_lr = old_lr * factor
            g['lr'] = new_lr
            old_lrs.append(old_lr)
            new_lrs.append(new_lr)

        # old_lrs = [g['lr'] for g in pgs]
        # for g in pgs:
        #     g['lr'] *= factor
        # new_lrs = [g['lr'] for g in pgs]

        return old_lrs, new_lrs

    def should_do_step(self, batch_idx):
        # Returns: bool, old_lrs to restore if needed
        # TODO:
        se = self.step_every
        do_step = (batch_idx % se) == (se-1)
        return do_step, None

        # old_lrs = None
        # if not do_step and (batch_idx == (num_batches - 1)):
        #     do_step = True
        #     factor = ((batch_idx % se) + 1) / se
        #     old_lrs, _ = self.scale_lr(factor)

        # return do_step, old_lrs

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
        # self.nag_with_predictor = nag_with_predictor # handled inside the wp.

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler
    
    def set_gap_aware(self, gap_aware):
        self.gap_aware = gap_aware

    def set_weight_stasher(self, weight_stasher: WeightStasher):
        if self.is_last_partition and not (weight_stasher is None):
            raise NotImplementedError()

        if self.is_problematic:
            is_forward = self.PROBLEMATIC_POLICY == 'SAME'
            weight_stasher.set_problematic(forward=is_forward)

        self.weight_stasher = weight_stasher

    def train(self):
        self.comm_handler.set_tensor_shapes(self.training_tensor_shapes)
        self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)

        self.partition.train()

        # Handles the transition : eval -> train

        if self.fwd_rcev_buffers.is_initialized():
            self.fwd_rcev_buffers.create()

    def eval(self):
        self.comm_handler.set_tensor_shapes(self.eval_tensor_shapes)
        # FIXME: we set eval dtypes as training too.
        self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)

        self.partition.eval()

        # Handles the transition : train -> eval
        if self.fwd_rcev_buffers.is_initialized():
            self.fwd_rcev_buffers.create()

        self.recv_all_bwd = True  # TODO: i don't like this hack

    def wait_on_sent_object(self, is_fwd):
        # TODO: can write the entire thing MUCH more nicely
        # if we just save asside and insert the new objects at the end.

        obj_holder = self.async_fwd_objects if is_fwd else self.async_bwd_objects

        # Pop the item that was increaced first.
        _, (sent_request_objects, tmp_sent_items) = obj_holder.popitem(
            last=False)
        for i in sent_request_objects:
            while(not i.is_completed()):
                pass

    def run_batch_forward(self, batch_idx, num_batches, done_bwds=None):
        """ Handles the forward pass, for last partition also handles the backward pass.

            Algorithem:
                - Get the data
                - Forward pass (including: wp, ws)
                    optional: Weight Preiction (wp)
                    optional: Weight Stashing (ws)
                - Send to next partition (*if needed)
                - If last partition: do the backward and send to previous partition

            TODO:
                -
        """
        # TODO: BIG_BATCH: fix delay in case micro batches.
        if self.gap_aware and self.partition.training:
            self.delay_at_batch[batch_idx] = self.expected_staleness(
                batch_idx, done_bwds)

        # Get the data to do forward on
        if self.is_first_partition:
            # data = next(self.dl_iter)
            # TODO: handle y with separate coordinated dataloader according to trainer/tast.
            x, *ctx = self.task.unpack_data_for_partition(next(self.dl_iter))
        else:
            if not self.fwd_rcev_buffers.is_initialized():
                self.fwd_rcev_buffers.create()

            recved_all = False
            if self.fwd_rcev_buffers.first_rcv_after_created or self.fwd_rcev_buffers.max_buffers == 1:
                self.fwd_rcev_buffers.recv_all(batch_idx, num_batches)
                recved_all = True

            x = self.fwd_rcev_buffers.wait_first()
            x = self.comm_handler.fix_after_recv(x)

            # pre-Start the next fwd Irecv:
            # TODO: decide if this is the best place to do it

            # This makes sure we don't overrun the buffer.
            # actually, many times we clone the input anyway inside the partition (for re-computation)
            # and if so, we can use less recv buffers for forward to save memory,
            # while stil getting the same speed/parallelism.
            if (not recved_all) and batch_idx - 1 + self.fwd_rcev_buffers.max_buffers < num_batches:
                self.fwd_rcev_buffers.recv_next(batch_idx-1)

            x, *ctx = self.task.unpack_data_for_partition(x)

        # Do the forward pass with optionals
        if self.weight_predictor and self.partition.training:
            # TODO: last partition can do bengio nesterov instead of predicting.
            # Requires per partition optimizer config, or some hack.
            self.weight_predictor.setup(
                self.expected_staleness(batch_idx, done_bwds))
            self.weight_predictor.forward()
            x = self.partition(x, batch_idx)

            if self.weight_stasher and self.partition.training:
                # Stash parameters for later.
                # Note: wait stasher should be None be in last partition.
                self.weight_stasher.stash_current(
                    batch_idx, self.expected_staleness(batch_idx, done_bwds))

            self.weight_predictor.revert()
        else:
            x = self.partition(x, batch_idx)
            if self.weight_stasher and self.partition.training:
                self.weight_stasher.stash_current(
                    batch_idx, self.expected_staleness(batch_idx, done_bwds))

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
            request_objects = self.comm_handler.send_gradients(
                grads, batch_idx)

            # BIG_BATCH allow skipping steps.
            # HACK: last partition- batch idx is the same as num backwards.
            do_step, old_lrs = self.should_do_step(batch_idx)

            # scale_down_lr = False
            if not do_step and (batch_idx == (num_batches - 1)):
                do_step = True
                # For the last batch, we must scale down the learning rate, and then restore.
                pgs = self.trainer.optimizer.param_groups
                old_lrs = [g['lr'] for g in pgs]
                for g in pgs:
                    # FIXME: micro batch index
                    g['lr'] *= ((batch_idx % self.step_every) +
                                1) / self.step_every

            # Step
            self.trainer.last_partition_step_and_statistics(
                x, *ctx, step_and_stats_ctx, step=do_step)
            del x, ctx, step_and_stats_ctx

            # Print training statistics.
            # if self.partition.training:
            self.batches += 1
            if self.batches % self.log_frequency == 0:
                batch_log_str = ''
                if hasattr(self.trainer, "scheduler"):
                    # Note: could be more than one LR, but we ignore this for simplicity.
                    lr = self.trainer.scheduler.get_last_lr()[0]
                    batch_log_str += '| lr {:02.4f}'.format(lr)

                # TODO: add more stats. e.g can print here time, ' ms/batch {:5.2f} | ' ,...
                self.logger.info(batch_log_str)

            if old_lrs:
                # return to previous LRs.
                for g, old_lr in zip(pgs, old_lrs):
                    g['lr'] = old_lr

        return request_objects

    def run_batch_backward(self, batch_idx, num_batches):
        """ Runs the backwards pass + step for all except the last partition """

        if not self.bwd_rcev_buffers.is_initialized():
            self.bwd_rcev_buffers.create()

        recved_all = False
        if self.recv_all_bwd or self.bwd_rcev_buffers.first_rcv_after_created or self.bwd_rcev_buffers.max_buffers == 1:
            self.recv_all_bwd = False
            self.bwd_rcev_buffers.recv_all(batch_idx, num_batches)
            recved_all = True

        #  Recompute before waiting to the first, so parallelize communication and computation
        self.partition.recompute(batch_idx)
        g = self.bwd_rcev_buffers.wait_first()
        g = self.comm_handler.fix_after_recv(g)

        # Wait for next if appropriate
        if (not recved_all) and batch_idx - 1 + self.bwd_rcev_buffers.max_buffers < num_batches:
            self.bwd_rcev_buffers.recv_next(batch_idx-1)

        # real_theta = None
        if self.weight_stasher:
            # self.weight_stasher.ensure_correct_post_restore(batch_idx)
            # Restore to parameters which the fwd ran on
            self.weight_stasher.pop_restore_stashed(batch_idx)
            # real_theta = self.weight_stasher.tmp_buff_top()

        # Compute gradeint
        self.partition.backward_from_recomputed(g, batch_idx)

        # Step and statistics
        request_objects = None
        if not (self.is_first_partition):
            g = self.partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)

        # BIG_BATCH allow skiping steps.
        do_step, old_lrs = self.should_do_step(batch_idx)
        # if self.stage == 0:
        #     print(f"do_step:{do_step}, step_every {self.step_every}, \
        #       bwd_batch_index:{batch_idx}, micro batch {batch_idx % self.step_every}, old_lrs:{old_lrs} ")

        # also do step for the last. (but with smaller LR)
        # scale_down_lr = False
        if not do_step and (batch_idx == (num_batches - 1)):
            do_step = True
            # TODO: For the last batch, we must scale down.
            # scale_down_lr = True
            pgs = self.trainer.optimizer.param_groups
            old_lrs = [g['lr'] for g in pgs]
            se = self.step_every
            for g in pgs:
                # FIXME: micro batch index
                g['lr'] *= ((batch_idx % se) + 1) / se

        if do_step:
            real_theta = self.weight_stasher.tmp_buff_top() if self.weight_stasher else None
            # ####### Preparing to step
            if real_theta:
                self.trainer.try_record_real_gap_from_current(real_theta)

            delay = self.delay_at_batch.pop(batch_idx, None)
            if not (delay is None) and self.is_problematic:
                # Average delays
                mb = self.get_micro_batch(batch_idx)
                if mb > 0:
                    delays = [
                        delay] + [self.delay_at_batch.pop(batch_idx-i, None) for i in range(1, mb + 1)]
                    delay = np.mean(delays)

            # possible Modify gradients (e.g Gap aware)
            self.trainer.modify_gradients(real_theta, delay)

            if self.weight_stasher:
                # Restore to previosly saved parameters, so we can do the step on them.
                self.weight_stasher.post_restore(batch_idx)
                # Mark previously stashed weights as dirty
                self.weight_stasher.mark_stashed_as_dirty()

            self.trainer.non_last_partition_step()

            if old_lrs:
                pgs = self.trainer.optimizer.param_groups
                for g, old_lr in zip(pgs, old_lrs):
                    g['lr'] = old_lr
        else:
            if self.weight_stasher:
                # Restore to previosly saved parameters, so we can do the next forwards on them!
                self.weight_stasher.post_restore_from_top(batch_idx)

        return request_objects

    def expected_staleness(self, done_fwds, done_bwds):
        # TODO: add batch considuration?
        # FFFFBFBFBFBFBFBFBFBFBFBFBBBB
        # Batch | bwds   | diff | staleness
        # 0     |  0     |   0  |    0   |

        # TODO: just pre compute a table in the beggining of the run based on this.
        # I don't care too much about the formula, there is probobly a nice one.
        # se = self.step_every
        # sem = se-1
        # return sum([x % se == sem for x in range(done_bwds, done_fwds)])

        return sum([self.should_do_step(x)[0] for x in range(done_bwds, done_fwds)])

    def expected_unseen_lrs(self, fwd_batch_index):
        # TODO:
        # (1) get expected_staleness
        # (2) ask the scheduler about the next x steps...
        # step_batches = [x for x in range(done_bwds, done_fwds) if self.should_do_step(x)]
        raise NotImplementedError()

    def run_forward_until_flush(self, num_batches):
        """
        Running evaluation (pipelined)
        Requires:
            set_dataloader() was called (if first partition)
            eval() was called
        """

        for done_fwds in range(num_batches):

            sent_request_objects = self.run_batch_forward(
                done_fwds, num_batches)
            if sent_request_objects:  # last partition returns empty list.
                self.async_fwd_objects[done_fwds] = sent_request_objects

            if len(self.async_fwd_objects) > 1:
                self.wait_on_sent_object(is_fwd=True)

        # Also clear in the end, just in case...
        while len(self.async_fwd_objects) > 0:
            self.wait_on_sent_object(is_fwd=True)

    def run_until_flush(self, num_batches, sched_step=True):
        """
        Requires:
            set_dataloader() was called (if first partition)
            train() was called
        """
        done_bwds = 0
        done_fwds = 0

        ga = self.gap_aware
        if ga:
            ga.skip_one_apply()

        # num_steps = num_batches
        while done_bwds < num_batches:
            # for step_index in range(num_steps):
            # Act according to some policy
            action_is_fwd = self.work_scheduler(self.stage, self.num_stages,
                                                num_batches, done_fwds, done_bwds)

            # if self.stage == 0:
            #     print(
            #         f"action_is_fwd:{action_is_fwd}. totla:{num_batches}, fwd:{done_fwds}, bwd:{done_bwds} ")

            if action_is_fwd:
                sent_request_objects = self.run_batch_forward(
                    done_fwds, num_batches, done_bwds)
                # Last partition inserts its gradints into async_fwd_objects,
                self.async_bwd_objects[done_fwds] = sent_request_objects
            else:
                sent_request_objects = self.run_batch_backward(
                    done_bwds, num_batches)
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
                self.wait_on_sent_object(is_fwd=True)

            if len(self.async_bwd_objects) > 1:
                self.wait_on_sent_object(is_fwd=False)

        # This print its used to debug how much we have in pipe.
        # print(self.stage, len(self.async_bwd_objects),
        #       len(self.async_fwd_objects))

        while len(self.async_fwd_objects) > 0:
            self.wait_on_sent_object(is_fwd=True)

        while len(self.async_bwd_objects) > 0:
            self.wait_on_sent_object(is_fwd=False)
        
        if sched_step:
            self.lr_scheduler.step()
            if ga:
                ga.update_max_lr()

        # if not self.comm_handler.cpu:
        #     # HACK: synchronize.
        #     torch.cuda.synchronize(device=self.device)
