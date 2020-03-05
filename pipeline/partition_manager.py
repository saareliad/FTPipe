import torch
import logging
from typing import Dict
from collections import OrderedDict
from . import CommunicationHandlerBase
from .partition import Partition, LastPartition, FirstPartition, PartitionWithoutRecomputation
from .partition import get_buffers_for_ddp_sync
from .training.interface import PartitionedTrainer
from .tasks import DLTask
from .weight_prediction.interface import WeightPredictor
from .gap_aware import GapAware  # TODO: change to interface.
from .work_schedulers import WorkScheduler, get_fwds_between_first_and_seconds_step_for_stage
from .weight_stashing import WeightStasher
from .buffer import Buffers
import numpy as np
import types
from .true_weights_storage import TrueWeightsStorage

# from gpu_mem_track import MemTracker
# import time


class SinglePartitionManager:
    PROBLEMATIC_POLICY = 'SAME'
    # PROBLEMATIC_POLICY = 'SKIP'

    def __init__(self, stage, num_stages, partition: torch.nn.Module,
                 comm_handler: CommunicationHandlerBase,
                 work_scheduler: WorkScheduler,
                 training_tensor_shapes, eval_tensor_shapes, training_tensor_dtypes,  # FIXME
                 device, is_last_partition, is_first_partition, log_frequency=100, max_buffers=2, step_every=1,
                 keep_buffers_alive=False, use_recomputation=True,
                 gap_aware_just_loss=False, sync_buffers=False,
                 ):

        if (gap_aware_just_loss and (not use_recomputation)):
            raise NotImplementedError(
                "gap_aware_just_loss works only with recomputation on")

        self.logger = logging.getLogger("msnag")  # FIXME

        self.gap_aware_just_loss = gap_aware_just_loss
        # TODO: work in progress, need to support exp name too, etc.
        self.weight_stashing_just_for_stats = False

        TO_DEVICE = False

        # Set partition.
        if use_recomputation:
            if is_last_partition:
                partition_cls = LastPartition
            elif is_first_partition:
                partition_cls = FirstPartition
            else:
                partition_cls = Partition
            self.partition = partition_cls(
                partition, device, to_device=TO_DEVICE)
        else:
            # Partition without recomputation
            if is_last_partition:
                partition_cls = LastPartition
                self.partition = partition_cls(
                    partition, device, to_device=TO_DEVICE)
            elif is_first_partition:
                partition_cls = PartitionWithoutRecomputation
                self.partition = partition_cls(
                    partition, device, to_device=TO_DEVICE, _REQ_GRAD=False)
            else:
                partition_cls = PartitionWithoutRecomputation
                self.partition = partition_cls(
                    partition, device, to_device=TO_DEVICE, _REQ_GRAD=True)

        if not TO_DEVICE:
            self.partition.to(device)

        self.comm_handler = comm_handler
        comm_handler.init_process_group()
        self.is_replicated = False
        self.sync_buffers = sync_buffers
        
        if hasattr(comm_handler, "init_ddp_context"):
            ddp = comm_handler.init_ddp_context(self.partition.layers)
            self.partition.layers = ddp
            self.is_replicated = True
            self.logger.info(
                f"Initialized DDP stage replication for for stage {stage}.")
            self.backward_nosync_context_manager = ddp.no_sync
            if sync_buffers:
                self.buffers_to_sync = get_buffers_for_ddp_sync(partition.layers)


        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition
        self.stage = stage
        self.num_stages = num_stages

        self.step_every = step_every
        self.work_scheduler = work_scheduler(step_every)

        # self.needs_own_post_restore = dict()

        # HACK:
        # FIXME: num batches just have to be high enough for the calculation.
        fwds, is_problematic = get_fwds_between_first_and_seconds_step_for_stage(
            self.work_scheduler, self.stage, self.num_stages, num_batches=390)
        self.is_problematic = is_problematic
        if is_problematic:
            print(
                f"-V- Patching problematic batches {fwds} for stage {self.stage}")
            if self.step_every > 2:
                raise NotImplementedError("check in shcedulers.")

            if self.PROBLEMATIC_POLICY == 'SKIP':
                self.set_problematic_skip()

        self.weight_predictor = None
        self.gap_aware = None
        self.weight_stasher = None

        # State for train logging
        self.log_frequency = log_frequency
        self.batches = 0

        # State for saving current relevant weight.
        self.true_weights_storage = None

        # State for recv buffers
        if max_buffers > 2:  # FIXME
            raise NotImplementedError()

        def make_buff(is_bwd, create=False):

            if is_bwd:
                b = Buffers(
                    max_buffers, self.comm_handler.create_gradients_rcv_buffers,
                    self.device, self.comm_handler.recv_gradients, is_grad=True)

            else:
                b = Buffers(
                    max_buffers, self.comm_handler.create_activations_recv_buffers,
                    self.device, self.comm_handler.recv_activations, is_grad=False)

            if create:
                b.create()
            return b

        shapes_are_equal = eval_tensor_shapes == training_tensor_shapes
        if shapes_are_equal:  # FIXME: also for eval dtypes.
            keep_buffers_alive = True  # HACK: if same shapes and datatypes, the buffers can remain!

        self.keep_buffers_alive = keep_buffers_alive

        if keep_buffers_alive:
            # Create once.

            # itertools.product(['fwd', 'bwd'], ['train', 'eval'])

            self.comm_handler.set_tensor_shapes(self.training_tensor_shapes)
            self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)

            self.fwd_rcev_buffers_train = make_buff(is_bwd=False, create=True)
            self.bwd_rcev_buffers = make_buff(is_bwd=True, create=False)

            self.comm_handler.set_tensor_shapes(self.eval_tensor_shapes)
            # FIXME: we set eval dtypes as training too.
            self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)
            if not shapes_are_equal:
                self.fwd_rcev_buffers_eval = make_buff(
                    is_bwd=False, create=True)
            else:
                self.fwd_rcev_buffers_eval = self.fwd_rcev_buffers_train  # HACK: same buffer!
        else:
            self.fwd_rcev_buffers = make_buff(is_bwd=False)
            self.bwd_rcev_buffers = make_buff(is_bwd=True)

        # Holds Async handle objects (for isends)
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

        # self.modify_gradients_before_send = False  # TODO add as option
        self.delay_at_batch = {}

        # Hints,May be set later.
        # self.dl_iter = None
        self.task: DLTask
        self.trainer: PartitionedTrainer
        self.weight_predictor: WeightPredictor
        self.gap_aware: GapAware
        self.weight_stasher: WeightStasher
        self.true_weights_storage: TrueWeightsStorage

    def set_true_weights_storage(self, true_weights_storage):
        self.true_weights_storage = true_weights_storage

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
                # FIXME: this funcion should not do the scale.
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

    def set_task(self, task: DLTask):
        self.task = task

    def set_trainer(self, trainer: PartitionedTrainer):
        self.trainer = trainer

    def set_dataloader(self, dataloader):
        assert self.is_first_partition or self.is_last_partition
        # self.dataloader = dataloader
        self.dl_iter = iter(dataloader)

    def set_weight_predictor(self, weight_predictor: WeightPredictor, nag_with_predictor: bool):
        self.weight_predictor = weight_predictor
        # self.nag_with_predictor = nag_with_predictor # handled inside the wp.

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_gap_aware(self, gap_aware):
        self.gap_aware = gap_aware

    def set_weight_stasher(self, weight_stasher: WeightStasher):
        assert (weight_stasher is not None)
        if self.is_last_partition:
            raise NotImplementedError()

        # if weight_stasher and self.weight_predictor and self.step_every > 1:
        #     weight_stasher.set_problematic(forward=True, policy='EVERY_BATCH')

        if self.is_problematic:
            if self.PROBLEMATIC_POLICY == 'SAME':
                if self.weight_predictor is None:
                    weight_stasher.set_problematic(
                        forward=True, policy='CHANGE')
                else:
                    weight_stasher.set_problematic(
                        forward=True, policy='EVERY_BATCH')
        elif self.PROBLEMATIC_POLICY == 'SKIP':
            raise NotImplementedError()
            # weight_stasher.set_problematic(forward=False, policy='CHANGE')

        self.weight_stasher = weight_stasher

    def train(self):
        self.comm_handler.set_tensor_shapes(self.training_tensor_shapes)
        self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)

        self.partition.train()

        # Handles the transition : eval -> train
        if self.keep_buffers_alive:
            self.fwd_rcev_buffers = self.fwd_rcev_buffers_train.reset_state()
            self.bwd_rcev_buffers.reset_state()
        else:
            if self.fwd_rcev_buffers.is_initialized():
                self.fwd_rcev_buffers.create()
            self.bwd_rcev_buffers.reset_state()

    def eval(self):
        self.comm_handler.set_tensor_shapes(self.eval_tensor_shapes)
        # FIXME: we set eval dtypes as training too.
        self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)

        self.partition.eval()

        # Handles the transition : train -> eval
        if self.keep_buffers_alive:
            self.fwd_rcev_buffers = self.fwd_rcev_buffers_eval.reset_state()
        else:
            if self.fwd_rcev_buffers.is_initialized():
                self.fwd_rcev_buffers.create()
        
        if self.is_replicated and self.sync_buffers:
            self.comm_handler.sync_buffers(self.buffers_to_sync)

    def wait_on_sent_object(self, is_fwd):
        # TODO: can write the entire thing MUCH more nicely
        # if we just save asside and insert the new objects at the end.

        obj_holder = self.async_fwd_objects if is_fwd else self.async_bwd_objects

        # Pop the item that was increaced first.
        _, (sent_request_objects, tmp_sent_items) = obj_holder.popitem(
            last=False)
        for i in sent_request_objects:
            i.wait()
            # while(not i.is_completed()):
            #     pass

    def get_input_data_forward(self, batch_idx, num_batches):

        # Get the data to do forward on
        if self.is_first_partition:
            # data = next(self.dl_iter)
            # TODO: handle y with separate coordinated dataloader according to trainer/tast.
            x, *ctx = self.task.unpack_data_for_partition(next(self.dl_iter))
            return x, ctx

        fwd_rcev_buffers = self.fwd_rcev_buffers

        if not fwd_rcev_buffers.is_initialized():
            fwd_rcev_buffers.create()

        recved_all = False
        if fwd_rcev_buffers.first_rcv_after_created or fwd_rcev_buffers.max_buffers == 1:
            fwd_rcev_buffers.recv_all(batch_idx, num_batches)
            recved_all = True

        x = fwd_rcev_buffers.wait_first()
        x = self.comm_handler.fix_after_recv(x)

        # pre-Start the next fwd Irecv:
        # TODO: decide if this is the best place to do it

        # This makes sure we don't overrun the buffer.
        # actually, many times we clone the input anyway inside the partition (for re-computation)
        # and if so, we can use less recv buffers for forward to save memory,
        # while stil getting the same speed/parallelism.
        if (not recved_all) and batch_idx - 1 + fwd_rcev_buffers.max_buffers < num_batches:
            fwd_rcev_buffers.recv_next(batch_idx-1)

        x, *ctx = self.task.unpack_data_for_partition(x)

        return x, ctx

    def forward_pass_and_send(self, batch_idx, num_batches):
        x, ctx = self.get_input_data_forward(batch_idx, num_batches)
        x = self.partition(x, batch_idx)
        request_objects = None
        if not self.is_last_partition:
            send_ctx = self.task.pack_send_context(x, *ctx)
            request_objects = self.comm_handler.send_activations(
                send_ctx, batch_idx)
        return request_objects, x, ctx

    def run_batch_forward(self, batch_idx, num_batches, done_bwds=None):
        """ Handles the forward pass, for last partition also handles the backward pass.

            Algorithem:
                - Get the data
                - Forward pass (including: wp, ws)
                    optional: Weight Preiction (wp)
                    optional: Weight Stashing (ws)
                - Send to next partition (*if needed)
                - If last partition: do the backward and send to previous partition


            In more detail:
                # (1) PRELOAD (do stuff like load wieghts, NAG, etc....)
                # (2) the actual forward
                # (3) send activation (if not last partition)
                # (4) stash weights if needed, etc. (NOTE: last partition don't stash)
                # (5) last partition does its thing:
                # (5.1) recompute
                # (5.2) send activation back
                # (5.3) resotre, step,...

            Feature:
                - Pre load Y to last partition if possible
        """
        partition = self.partition
        is_training = partition.training

        if is_training:
            expected_staleness = self.expected_staleness(batch_idx, done_bwds)
            self.delay_at_batch[batch_idx] = expected_staleness

        # Get the data to run forward on, (and target)
        if self.is_last_partition:
            preload_ctx = self.task.preload_last_partition(
                getattr(self, "dl_iter", None), self.device)

        # Do the forward pass with optionals
        # optional (1): Weight Prediction
        # optional (2): Weight Stashing
        if is_training:
            weight_predictor = self.weight_predictor
            weight_stasher = self.weight_stasher
            if weight_predictor is not None:
                # TODO: last partition can do bengio nesterov instead of predicting.
                # Requires per partition optimizer config, or some hack.

                # NOTE: (1) we scale LR here just to tell weight predictor. will do it again when we step.
                # NOTE: (2) true_weights_storage stuff handled inside predictor.
                old_lrs = None
                if batch_idx >= self.first_effected_batch:
                    old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)

                weight_predictor.setup(expected_staleness)
                weight_predictor.forward()
                # Moved by: sum(i.norm() for i in self.true_weights_storage.true_weights[0])
                # - sum([i.norm() for i in self.trainer.optimizer.param_groups[0]['params']])
                if old_lrs:
                    pgs = self.trainer.optimizer.param_groups
                    for pg, old_lr in zip(pgs, old_lrs):
                        pg['lr'] = old_lr

                request_objects, x, ctx = self.forward_pass_and_send(
                    batch_idx, num_batches)

                if weight_stasher is not None:
                    # Stash parameters for later.
                    # Note: wait stasher should be None be in last partition.

                    # TODO: option to do it in all except last partition.  ("NAG ONLY STALENESS 0")
                    # This is only one batch per epoch, so it does not really matter.
                    if expected_staleness == 0 and weight_predictor.nag_with_predictor:
                        expected_staleness = 1
                        # HACK: apparently, no reason to stash, so why we do it here? so we can reload.

                    # HACK: will set dirty ahead!
                    weight_stasher.stash_current(
                        batch_idx, expected_staleness)

                # HACK: will revert after send.
                # weight_predictor.revert()
            else:
                # No weight predictor
                request_objects, x, ctx = self.forward_pass_and_send(
                    batch_idx, num_batches)
                if weight_stasher is not None:
                    weight_stasher.stash_current(batch_idx, expected_staleness)
        else:
            # Not training. just go on as usual
            request_objects, x, ctx = self.forward_pass_and_send(
                batch_idx, num_batches)

        if not self.is_last_partition:
            # For the last partition - we restore later.
            if is_training:
                self.true_weights_storage.restore_if_needed()
            return request_objects

        else:
            # Last partition - also do backward.
            ctx = (*preload_ctx, *ctx)
            if not is_training:
                # In Eval: Just calculate statistics.
                self.trainer.calc_test_stats(x, *ctx)
                return []

            trainer = self.trainer

            # NOTE: for last partition- batch idx is the same as num backwards.
            do_step, old_lrs = self.should_do_step(batch_idx)
            # Backprop
            # For the last batch, we must scale down the learning rate, and then restore.
            if (not do_step) and (batch_idx == (num_batches - 1)):
                do_step = True
                old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)

            if (not do_step) and self.is_replicated:
                with self.backward_nosync_context_manager():
                    step_and_stats_ctx = trainer.backprop_last_partition(
                        x, *ctx)
            else:
                step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)

            # Send partition border gradients
            grads = partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(
                grads, batch_idx)

            self.true_weights_storage.restore_if_needed()  # check=False

            # Step
            trainer.last_partition_step_and_statistics(
                x, *ctx, step_and_stats_ctx, step=do_step)

            if do_step:
                self.true_weights_storage.reset_on_step()

            # Print training statistics.
            self.batches += 1
            if self.batches % self.log_frequency == 0:
                batch_log_str = ''
                if hasattr(trainer, "scheduler"):
                    # Note: could be more than one LR, but we ignore this for simplicity.
                    lr = trainer.scheduler.get_last_lr()[0]
                    batch_log_str += '| lr {:02.4f}'.format(lr)

                # TODO: add more stats. e.g can print here time, ' ms/batch {:5.2f} | ' ,...
                self.logger.info(batch_log_str)

            if old_lrs:
                # return to previous LRs.
                pgs = trainer.optimizer.param_groups
                for g, old_lr in zip(pgs, old_lrs):
                    g['lr'] = old_lr

        return request_objects

    def run_batch_backward(self, batch_idx, num_batches):
        """ Runs the backwards pass + step for all except the last partition """

        bwd_rcev_buffers = self.bwd_rcev_buffers
        if not bwd_rcev_buffers.is_initialized():
            bwd_rcev_buffers.create()

        recved_all = False
        if bwd_rcev_buffers.first_rcv_after_created or bwd_rcev_buffers.max_buffers == 1:
            bwd_rcev_buffers.recv_all(batch_idx, num_batches)
            recved_all = True

        weight_stasher = self.weight_stasher
        #  Recompute before waiting to the first, so parallelize communication and computation
        if weight_stasher and (not self.gap_aware_just_loss):
            # Restore to parameters which the fwd ran on
            weight_stasher.pop_restore_stashed(batch_idx)
            # self.true_weights_storage.record_change_mode("stashed")

        # elif (self.weight_predictor and self.weight_predictor.nag_with_predictor
        #       and (self.delay_at_batch.get(batch_idx) == 0)):
        #     # HACK: feature: do nag in for the backward pass.
        #     self.weight_predictor.setup(0)
        #     self.weight_predictor.forward()
        #     # self.true_weights_storage.record_change_mode("pred")

        self.partition.recompute(batch_idx)

        g = bwd_rcev_buffers.wait_first()
        g = self.comm_handler.fix_after_recv(g)

        # Allow skiping steps (Gradient aggregation)
        do_step, old_lrs = self.should_do_step(batch_idx)

        # also do step for the last. (but with smaller LR)
        if not do_step and (batch_idx == (num_batches - 1)):
            do_step = True
            old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)

        # Compute gradeints
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                self.partition.backward_from_recomputed(g, batch_idx)
        else:
            self.partition.backward_from_recomputed(g, batch_idx)

        # recompute and send backward
        request_objects = None
        if not (self.is_first_partition):
            g = self.partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)

        # Wait for next if appropriate
        if (not recved_all) and batch_idx - 1 + bwd_rcev_buffers.max_buffers < num_batches:
            bwd_rcev_buffers.recv_next(batch_idx-1)

        if do_step:
            trainer = self.trainer
            weight_stasher = self.weight_stasher
            # TODO: allow access to real theta just for statistics
            if weight_stasher:
                if self.gap_aware_just_loss:
                    stashed_theta = weight_stasher.pop_stashed_buff(batch_idx)
                    trainer.try_record_real_gap_from_current(stashed_theta)
                    real_theta = None
                else:
                    real_theta = self.true_weights_storage.get_true_weights()
                    trainer.try_record_real_gap_from_current(real_theta)
                    stashed_theta = None
            else:
                real_theta = None
                stashed_theta = None
            # ####### Preparing to step

            if self.gap_aware:
                # Get delay and modify gradeints.
                if self.is_problematic:
                    # Average delays
                    mb = self.get_micro_batch(batch_idx)
                    delay = np.mean([self.delay_at_batch.pop(
                        batch_idx-i) for i in range(0, mb + 1)])
                else:
                    delay = self.delay_at_batch.pop(batch_idx)

                # Modify gradients
                trainer.modify_gradients(
                    real_theta=real_theta, delay=delay, stashed_theta=stashed_theta)

            if weight_stasher:
                # Mark previously stashed weights as dirty
                weight_stasher.mark_stashed_as_dirty()

            # Restore to previosly saved parameters, so we can do the step on them.
            self.true_weights_storage.restore_if_needed()
            self.true_weights_storage.reset_on_step()
            trainer.non_last_partition_step()

            if old_lrs:
                # Note that sometimes its not defined locally.
                pgs = trainer.optimizer.param_groups
                for g, old_lr in zip(pgs, old_lrs):
                    g['lr'] = old_lr
        else:
            self.true_weights_storage.restore_if_needed()
            # FIXME: probobly should be removed...
            if self.gap_aware_just_loss and self.weight_stasher:
                weight_stasher.pop_stashed_buff(batch_idx)

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
        # FIXME: for step_every > roundtrip.
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

        async_fwd_objects = self.async_fwd_objects
        wait_on_sent_object = self.wait_on_sent_object
        run_batch_forward = self.run_batch_forward

        for done_fwds in range(num_batches):
            sent_request_objects = run_batch_forward(
                done_fwds, num_batches)
            if sent_request_objects:  # last partition returns empty list.
                if async_fwd_objects:
                    wait_on_sent_object(is_fwd=True)
                async_fwd_objects[done_fwds] = sent_request_objects

        # Also clear in the end, just in case...
        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True)

    def run_until_flush(self, num_batches, sched_step=True):
        """
        Requires:
            set_dataloader() was called (if first partition)
            train() was called
        """
        done_bwds = 0
        done_fwds = 0

        reminder = num_batches % self.step_every
        # HACK: batch_idx always less than num batches
        self.first_effected_batch = num_batches - reminder
        self.reminder_scaler_lr_factor = reminder / self.step_every

        stage = self.stage
        num_stages = self.num_stages
        work_scheduler = self.work_scheduler
        is_first_partition = self.is_first_partition
        is_last_partition = self.is_last_partition
        run_batch_backward = self.run_batch_backward
        run_batch_forward = self.run_batch_forward
        async_bwd_objects = self.async_bwd_objects
        async_fwd_objects = self.async_fwd_objects
        wait_on_sent_object = self.wait_on_sent_object

        while done_bwds < num_batches:
            # Act according to some policy
            action_is_fwd = work_scheduler(stage, num_stages,
                                           num_batches, done_fwds, done_bwds)
            if action_is_fwd:
                sent_request_objects = run_batch_forward(
                    done_fwds, num_batches, done_bwds=done_bwds)
                # NOTE: Last partition inserts its gradints into async_fwd_objects,
                # wait on prev send
                if async_fwd_objects:
                    wait_on_sent_object(is_fwd=True)
                async_fwd_objects[done_fwds] = sent_request_objects
            else:
                sent_request_objects = run_batch_backward(
                    done_bwds, num_batches)
                if not (is_first_partition):
                    # wait on prev send
                    if async_bwd_objects:
                        wait_on_sent_object(is_fwd=False)
                    async_bwd_objects[done_bwds] = sent_request_objects

            # Increase counters
            if is_last_partition:
                done_bwds += 1
                done_fwds += 1
            else:
                if action_is_fwd:
                    done_fwds += 1
                else:
                    done_bwds += 1

        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True)

        while len(async_bwd_objects) > 0:
            wait_on_sent_object(is_fwd=False)

        if sched_step:
            self.lr_scheduler.step()
            if self.gap_aware:
                self.gap_aware.update_max_lr()
