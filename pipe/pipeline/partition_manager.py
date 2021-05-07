import logging
import os
import types
import warnings

import torch
from torch import Tensor
from tqdm import tqdm

from . import CommunicationHandlerBase
from .data_propagation import PipelineDataPropagator
from .gap_aware import GapAwareBase
from .partition import (GPipePartition, GPipeFirstPartition,
                        GPipeLastPartition)
from .partition import (Partition, LastPartition, FirstPartition,
                        PartitionWithoutRecomputation,
                        )
from .partition import get_buffers_for_ddp_sync
from .trainers import PipelineSupportedTrainerType
from .trainers.interface import MultiPartitionTrainer  # , GradNormStepperMixin
from .trainers.statistics.gap import try_record_real_gap_from_current
from .true_weights_storage import TrueWeightsStorage
from .weight_prediction.interface import WeightPredictor
from .weight_stashing import WeightStasher
from .work_schedulers import WorkScheduler

DEBUG_FAKE_DRAW = os.environ.get("DEBUG_FAKE_DRAW", False)


# TODO: multiprocessing: problem when sharing weights (sending nn.Parameters),
# TODO: weight prediction + weight stashing + step_every + cache (aggmsnag, no aggmsnag)
# TODO: gap aware + step_every >1
# TODO: currently assuming is_last_partition also means zero staleness.
# TODO: Mixed-Recomputation: partition class for non last stage which does not need to recompute (dictated by partitioning)
# TODO: Replicated stages: https://github.com/saareliad/async_msnag_pipeline/commit/dd699fb9f4df5b211c5d6a3fb821e9edeb699ecf
# TODO: consider making a STEP function decoupled from backward
# TODO: consider spliting the methods according to features (prediction / stashing / gap aware)

# experimental "virtual last partition"

class SinglePartitionManager:
    def __init__(self,
                 stage: int,
                 stage_depth: int,
                 pipeline_depth: int,
                 num_stages,
                 partition: torch.nn.Module,
                 comm_handler: CommunicationHandlerBase,
                 work_scheduler: WorkScheduler,
                 device,
                 is_last_partition,
                 is_first_partition,
                 log_frequency=100,
                 step_every=1,
                 use_recomputation=True,
                 gap_aware_just_loss=False,
                 sync_buffers=False,
                 weight_stashing_just_for_stats=False,
                 disable_clone_inputs=False,
                 req_grad=None,
                 scale_down_lr_for_smaller_batches=False
                 # outputs_req_grad=None,
                 ):

        if not disable_clone_inputs:
            disable_clone_inputs = True
            warnings.warn("setting disable_clone_inputs=True to avoid double clone since we clone in MPI too.")

        if gap_aware_just_loss and not use_recomputation:
            raise NotImplementedError(
                "gap_aware_just_loss works only with recomputation on")

        self.work_scheduler = work_scheduler
        self.logger = logging.getLogger("msnag")
        self.comm_handler = comm_handler
        self.sync_buffers = sync_buffers
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition or stage_depth == pipeline_depth - 1
        self.stage = stage

        self.pipeline_depth = pipeline_depth
        self.num_stages = num_stages
        self.step_every = step_every

        self.true_stage_depth = stage_depth
        if hasattr(self.work_scheduler, "get_virtual_stage_depth"):
            self.stage_depth = self.work_scheduler.get_virtual_stage_depth(stage_depth)
        else:
            self.stage_depth = stage_depth


        self._init_partition(partition, use_recomputation, disable_clone_inputs, req_grad)
        self.is_replicated = self._maybe_init_ddp(comm_handler, partition, stage, sync_buffers)
        self.comm_handler.init_buffers()
        self.futures_handler = self.comm_handler.create_futures_handler()

        self.weight_predictor = None
        self.gap_aware = None
        self.weight_stasher = None
        self.gap_aware_just_loss = gap_aware_just_loss
        self.weight_stashing_just_for_stats = weight_stashing_just_for_stats

        self.true_weights_storage = None
        self.delay_at_batch = {}
        self.saved_for_backward = dict()
        self.dl_iter = None

        # State for train logging
        self.log_frequency = log_frequency
        self.batches = 0

        # State for saving current relevant weight.
        self.true_weights_storage = None
        self.delay_at_batch = {}
        self.saved_for_backward = dict()
        self.dl_iter = None

        # Hints,May be set later.
        # self.dl_iter = None
        self.data_propagator: PipelineDataPropagator
        self.trainer: MultiPartitionTrainer
        self.weight_predictor: WeightPredictor
        self.gap_aware: GapAwareBase
        self.weight_stasher: WeightStasher
        self.true_weights_storage: TrueWeightsStorage

    def _maybe_init_ddp(self, comm_handler, partition, stage, sync_buffers):
        is_replicated = False
        if hasattr(comm_handler, "init_ddp_context"):
            is_replicated = True
            ddp = comm_handler.init_ddp_context(self.partition.layers)
            self.partition.layers = ddp
            self.logger.info(
                f"Initialized DDP stage replication for for stage {stage}.")
            self.backward_nosync_context_manager = ddp.no_sync
            if sync_buffers:
                self.buffers_to_sync = get_buffers_for_ddp_sync(
                    partition.layers)

        return is_replicated

    def _init_partition(self, partition, use_recomputation, disable_clone_inputs, req_grad,
                        ):

        if self.stage_depth == 0:
            use_recomputation = False

        TO_DEVICE = False
        is_last_partition = self.is_last_partition
        is_first_partition = self.is_first_partition

        device = self.device

        # Set partition.
        if use_recomputation:
            if is_last_partition:
                partition_cls = LastPartition
            elif is_first_partition:
                partition_cls = FirstPartition
            else:
                partition_cls = Partition
            self.partition = partition_cls(partition,
                                           device,
                                           to_device=TO_DEVICE,
                                           req_grad=req_grad, )
        else:
            # Partition without recomputation
            if is_last_partition:
                partition_cls = LastPartition
                self.partition = partition_cls(
                    partition,
                    device,
                    to_device=TO_DEVICE,
                    req_grad=req_grad,
                )
            elif is_first_partition:
                partition_cls = PartitionWithoutRecomputation
                self.partition = partition_cls(
                    partition,
                    device,
                    to_device=TO_DEVICE,
                    _REQ_GRAD=False,
                    req_grad=req_grad,
                )
            else:
                partition_cls = PartitionWithoutRecomputation
                self.partition = partition_cls(
                    partition,
                    device,
                    to_device=TO_DEVICE,
                    _REQ_GRAD=True,
                    req_grad=req_grad,
                )
        if disable_clone_inputs:
            # We do the clone ourself.
            # if hasattr(partition_cls, "_CLONE_INPUTS"):
            partition_cls._CLONE_INPUTS = False
        if not TO_DEVICE:
            self.partition.to(device)

    def set_true_weights_storage(self, true_weights_storage):
        self.true_weights_storage = true_weights_storage
        se = self.step_every

        def _get_micro_batch(self, batch_index):
            if batch_index <= se:
                return batch_index
            return (batch_index + 1) % se

        self.get_micro_batch = types.MethodType(_get_micro_batch, self)

    def get_micro_batch(self, batch_index):
        return batch_index % self.step_every

    def scale_lr(self, factor):
        pgs = self.trainer.optimizer.param_groups
        old_lrs = []
        for g in pgs:
            old_lr = g['lr']
            new_lr = old_lr * factor
            g['lr'] = new_lr
            old_lrs.append(old_lr)
        return old_lrs

    def is_last_micro_batch(self, batch_idx) -> bool:
        """Simply return if a this is the last micro batch"""
        se = self.step_every
        do_step = (batch_idx % se) == (se - 1)
        return do_step

    def set_data_propagator(self, data_propagator: PipelineDataPropagator):
        self.data_propagator = data_propagator

    def set_trainer(self, trainer: PipelineSupportedTrainerType):
        self.trainer = trainer

    def set_dataloader(self,
                       dataloader,
                       debug_run_limit=-1,
                       fake_draw=DEBUG_FAKE_DRAW,
                       dl_iter=None):
        if dl_iter is not None:
            self.dl_iter = dl_iter
        else:
            self.dl_iter = iter(dataloader)

            # FOR DEBUG
            if fake_draw and debug_run_limit > 0 and (debug_run_limit < len(dataloader)):
                fake_draw = len(dataloader) - debug_run_limit
                for _ in range(fake_draw):
                    next(self.dl_iter)

    def set_weight_predictor(self, weight_predictor: WeightPredictor):
        self.weight_predictor = weight_predictor

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_gap_aware(self, gap_aware):
        self.gap_aware = gap_aware
        if self.step_every > 1:
            raise NotImplementedError("deprecated, work in progress")

    def set_weight_stasher(self, weight_stasher: WeightStasher):
        assert (weight_stasher is not None)
        if self.is_last_partition:
            raise NotImplementedError("Assuming last stage does not need stashing")
        self.weight_stasher = weight_stasher

    def train(self):
        """Sets training mode.
            Also Handles the transition : eval -> train 
        """
        self.partition.train()
        self.comm_handler.train()

    def eval(self):
        """Sets evaluation mode.
            Also handles the transition : train -> eval
            Also handles buffer sync in case stage_id is replicated
        """
        self.comm_handler.eval()
        self.partition.eval()
        if self.is_replicated and self.sync_buffers:
            self.comm_handler.sync_buffers(self.buffers_to_sync)

    def maybe_log_lr(self):
        # Print training statistics.
        self.batches += 1
        if self.batches % self.log_frequency == 0:
            batch_log_str = ''
            if hasattr(self.trainer, "scheduler"):
                # Note: could be more than one LR, but we ignore this for simplicity.
                lr = self.trainer.scheduler.get_last_lr()[0]
                batch_log_str += '| lr {:02.9f}'.format(lr)
            self.logger.info(batch_log_str)

    def forward_pass_and_send(self,
                              batch_idx,
                              num_batches,
                              preload_input_partition):

        # Get data
        if self.is_first_partition:
            x = preload_input_partition
        else:
            # Get input data from previous pipeline stage
            last_due_end = batch_idx + 1 == num_batches
            x = self.comm_handler.get_data_forward(batch_idx, num_batches, last_due_end)
            # Unify with preloaded data
            x = (*preload_input_partition, *x)
        # In case we send labels in pipeline: extract them from the output.
        # For last partition: what is loaded for outside loss and statistics (e.g: batch size, ...)
        x, *ctx = self.data_propagator.unpack_data_for_partition(x)
        # Run the stage
        x = self.partition(x, batch_idx)
        request_objects = None
        # For non last partition - send forward.
        if not self.is_last_partition or self.true_stage_depth > 0:
            send_ctx = self.data_propagator.pack_send_context(x, *ctx)
            request_objects = self.comm_handler.send_activations(
                send_ctx, batch_idx)
        return request_objects, x, ctx

    def run_batch_forward(self, batch_idx, num_batches, done_bwds=None):
        """ Handles the forward pass, for last partition also handles the backward pass.

            Algorithm:
                - Get the data
                - Forward pass (including: wp, ws)
                    optional: Weight Prediction (wp)
                    optional: Weight Stashing (ws)
                - Send to next partition (*if needed)
                - If last partition: do the backward and send to previous partition


            In more detail:
                # (1) PRELOAD (do stuff like load weights, NAG, etc....)
                # (2) the actual forward
                # (3) send activation (if not last partition)
                # (4) stash weights if needed, etc. (NOTE: last partition don't stash)

                # (5) last partition does its thing:
                # (5.1) recompute
                # (5.2) send activation back
                # (5.3) restore, step,...

            Feature:
                - Pre load Y to last partition if possible
        """
        # preload stuff from dataloader.
        preload_input_partition, preload_input_to_outside_loss = self.data_propagator.preload_from_dataloader(
            self.dl_iter)

        partition = self.partition
        is_training = partition.training
        # Do the forward pass with optionals
        # optional (1): Weight Prediction
        # optional (2): Weight Stashing
        if is_training:
            expected_staleness = self.expected_staleness(batch_idx, done_bwds)
            self.delay_at_batch[batch_idx] = expected_staleness
            weight_predictor = self.weight_predictor
            weight_stasher = self.weight_stasher
            if weight_predictor is not None:
                # Requires per partition optimizer config, or some hack.

                # NOTE: (1) we scale LR here just to tell weight predictor. will do it again when we step.
                # NOTE: (2) true_weights_storage stuff handled inside predictor.
                old_lrs = None
                if batch_idx >= self.first_effected_batch:
                    old_lrs = self.scale_lr(self.reminder_scaler_lr_factor)

                weight_predictor.setup(expected_staleness)
                weight_predictor.forward()
                if old_lrs:
                    pgs = self.trainer.optimizer.param_groups
                    for pg, old_lr in zip(pgs, old_lrs):
                        pg['lr'] = old_lr

                request_objects, x, ctx = self.forward_pass_and_send(
                    batch_idx, num_batches, preload_input_partition)

                if weight_stasher is not None:
                    # Stash parameters for later.
                    # Note: wait stasher should be None be in last partition.

                    # This is only one batch per epoch, so it does not really matter.
                    if expected_staleness == 0 and weight_predictor.nag_with_predictor:
                        expected_staleness = 1
                        # HACK: apparently, no reason to stash, so why we do it here? so we can reload.

                    # HACK: will set dirty ahead!
                    weight_stasher.stash_current(batch_idx, expected_staleness)

                # HACK: will revert after send.
                # weight_predictor.revert()
            else:
                # No weight predictor
                request_objects, x, ctx = self.forward_pass_and_send(
                    batch_idx, num_batches, preload_input_partition)
                if weight_stasher is not None:
                    weight_stasher.stash_current(batch_idx, expected_staleness)

            if expected_staleness > 0 or self.true_stage_depth > 0:  # self.true_stage_depth > 0:  # Non zero staleness partition
                self.true_weights_storage.restore_if_needed()  # TODO: not ALWAYS the ideal place
                return request_objects

        else:
            # Eval:
            request_objects, x, ctx = self.forward_pass_and_send(
                batch_idx, num_batches, preload_input_partition)
            if self.is_last_partition:
                ctx = (*preload_input_to_outside_loss, *ctx)
                self.trainer.calc_test_stats(x, *ctx)
                return []
            else:
                return request_objects

        # Last partition: backward and step

        assert is_training
        assert self.true_stage_depth == 0, self.true_stage_depth

        ctx = (*preload_input_to_outside_loss, *ctx)
        self.saved_for_backward[batch_idx] = (x, *ctx)

        request_objects = self.last_partition_batch_backward(batch_idx, num_batches)
        return request_objects

    def last_partition_batch_backward(self, batch_idx: int, num_batches: int):
        # TODO: currently only last partition should be at depth 0.
        if not self.is_last_partition:
            raise NotImplementedError("currently only last partition should be at depth 0.")
        ############################
        # zero staleness partition backward
        # Last partition backward
        ############################
        # Last partition - also do backward.
        x, *ctx = self.saved_for_backward.pop(batch_idx)

        trainer = self.trainer
        # NOTE: for last partition- batch idx is the same as num backwards.
        old_lrs = None
        do_step = self.is_last_micro_batch(batch_idx)
        # Backprop
        last_due_end = batch_idx + 1 == num_batches
        if (not do_step) and last_due_end:
            # For the last batch, we must scale down the learning rate, and then restore.
            # Because the "step_every" policy: we won't usually step,
            # but since its the last batch - we just scale down LR and take a smaller step.
            # TODO: ability to run it off
            do_step = True
            old_lrs = self.scale_lr(self.reminder_scaler_lr_factor)
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)
        else:
            step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)  # NOTE: Usually, ctx is loss
        # Send partition border gradients
        request_objects = self.comm_handler.send_gradients(self.partition.get_grad(batch_idx), batch_idx)
        self.true_weights_storage.restore_if_needed()  # check=False
        # Step
        trainer.last_partition_step_and_statistics(x,
                                                   *ctx,
                                                   step_and_stats_ctx,
                                                   step=do_step,
                                                   old_lrs=old_lrs)
        if do_step:
            self.true_weights_storage.reset_on_step()
        self.maybe_log_lr()
        return request_objects

    def run_batch_backward(self, batch_idx, num_batches, next_backward_batch_idx=None):
        """ Runs the backwards pass + step for all except the last partition """
        last_due_end = batch_idx + 1 == num_batches
        self.comm_handler.pre_recv_gradients(batch_idx, num_batches, last_due_end)

        weight_stasher = self.weight_stasher
        #  Recompute before waiting to the first, so parallelize communication and computation
        if weight_stasher and (not self.gap_aware_just_loss):
            # Restore to parameters which the fwd ran on
            weight_stasher.pop_and_load_stashed_params(batch_idx)

        self.partition.recompute(batch_idx)
        # NOTE: in MPI version there was hacky zero and sync here
        g = self.comm_handler.wait_recv_gradients(batch_idx, last_due_end)

        # recv the next gradients.
        if next_backward_batch_idx is not None:
            next_backward_batch_idx_last_due_end = next_backward_batch_idx + 1 == num_batches
            self.comm_handler.pre_recv_gradients(next_backward_batch_idx, num_batches,
                                                 next_backward_batch_idx_last_due_end)

        # Allow skipping steps (Gradient aggregation)
        old_lrs = None
        do_step = self.is_last_micro_batch(batch_idx)
        # also do step for the last. (but with smaller LR)
        if not do_step and last_due_end:
            do_step = True
            old_lrs = self.scale_lr(self.reminder_scaler_lr_factor)

        # Compute gradients
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                self.partition.backward_from_recomputed(g, batch_idx)
        else:
            self.partition.backward_from_recomputed(g, batch_idx)

        # recompute and send backward
        request_objects = None
        if not self.is_first_partition:
            request_objects = self.comm_handler.send_gradients(
                self.partition.get_grad(batch_idx), batch_idx)

        # NOTE: we can start the next recv here
        if do_step:
            trainer = self.trainer
            weight_stasher = self.weight_stasher

            # allow access to real theta for statistics
            if weight_stasher:
                if self.gap_aware_just_loss or self.weight_stashing_just_for_stats:
                    stashed_theta = weight_stasher.pop_stashed_buff(batch_idx)
                    real_theta = None
                    not_loaded_theta = stashed_theta
                else:
                    real_theta = self.true_weights_storage.get_true_weights()
                    stashed_theta = None
                    not_loaded_theta = real_theta

                # NOTE we can get the gap for free from gap aware sometimes.
                try_record_real_gap_from_current(trainer.statistics, trainer.optimizer, not_loaded_theta,
                                                 pre_computed_gap=None)
            else:
                real_theta = None
                stashed_theta = None
            # ####### Preparing to step

            if self.gap_aware:
                # Get delay and modify gradients.
                if self.step_every > 1:
                    raise NotImplementedError()  # TODO:

                delay = self.delay_at_batch.pop(batch_idx)

                # Modify gradients
                # NOTE: can handle grad clip here instead of in step.
                trainer.apply_gap_aware(real_theta=real_theta,
                                        delay=delay,
                                        stashed_theta=stashed_theta)
            if weight_stasher:
                # Mark previously stashed weights as dirty
                weight_stasher.mark_stashed_as_dirty()

            # Restore to previously saved parameters, so we can do the step on them.
            self.true_weights_storage.restore_if_needed()
            self.true_weights_storage.reset_on_step()
            trainer.non_last_partition_step(old_lrs)

        else:
            self.true_weights_storage.restore_if_needed()
            # FIXME: probably should be removed...
            if self.gap_aware_just_loss and self.weight_stasher:
                weight_stasher.pop_stashed_buff(batch_idx)

        return request_objects

    def expected_staleness(self, done_fwds, done_bwds):
        # FFFFBFBFBFBFBFBFBFBFBFBFBBBB
        # Batch | bwds   | diff | staleness
        # 0     |  0     |   0  |    0   |
        # TODO: just pre compute a table in the beginning of the run based on this.
        # I don't care too much about the formula
        return sum([self.is_last_micro_batch(x) for x in range(done_bwds, done_fwds)])

    def run_forward_until_flush(self, num_batches):
        """
        Running evaluation (pipelined)
        Requires:
            set_dataloader() was called (in case stage requires data loading)
            eval() was called
        """

        run_batch_forward = self.run_batch_forward
        futures_handler = self.futures_handler

        if self.is_last_partition:
            b_tqdm = tqdm(range(num_batches), desc="Eval")
        else:
            b_tqdm = range(num_batches)

        for done_fwds in b_tqdm:
            ro = run_batch_forward(done_fwds, num_batches)
            futures_handler.after_forward(ro, done_fwds, False)
        futures_handler.clean_eval()

    def run_until_flush(self, num_batches, flush_rate=-1):
        """
        Requires:
            set_dataloader() was called (in case stage requires data loading)
            train() was called
        """
        done_bwds = 0
        done_fwds = 0

        reminder = num_batches % self.step_every
        # HACK: batch_idx always less than num batches
        self.first_effected_batch = num_batches - reminder
        self.reminder_scaler_lr_factor = reminder / self.step_every

        stage_depth = self.stage_depth
        true_stage_depth = self.true_stage_depth
        pipeline_depth = self.pipeline_depth

        work_scheduler = self.work_scheduler
        is_last_partition = self.is_last_partition
        run_batch_backward = self.run_batch_backward
        run_batch_forward = self.run_batch_forward
        futures_handler = self.futures_handler

        # sets warmup state for PipeDream. Else no-op.
        self.work_scheduler.reset()

        if is_last_partition:
            b_tqdm = tqdm(range(num_batches), desc="Train")
            b_tqdm_it = iter(b_tqdm)

        while done_bwds < num_batches:
            # Act according to some policy
            action_is_fwd = work_scheduler(stage_depth, pipeline_depth, num_batches,
                                           done_fwds, done_bwds)
            if action_is_fwd:
                ro = run_batch_forward(done_fwds,
                                       num_batches,
                                       done_bwds=done_bwds)
                if true_stage_depth == 0:
                    futures_handler.after_backward(ro, done_bwds)
                elif stage_depth == 0:
                    futures_handler.after_forward(ro, done_fwds, True)
                    ro = run_batch_backward(done_bwds, num_batches)
                    futures_handler.after_backward(ro, done_bwds)
                else:
                    futures_handler.after_forward(ro, done_fwds, True)

            else:

                if done_bwds + 1 < num_batches:
                    next_backward_batch_idx_to_run = done_bwds + 1
                else:
                    next_backward_batch_idx_to_run = None

                ro = run_batch_backward(done_bwds, num_batches, next_backward_batch_idx=next_backward_batch_idx_to_run)
                futures_handler.after_backward(ro, done_bwds)

            # Increase counters
            if stage_depth == 0:
                done_bwds += 1
                done_fwds += 1
                if is_last_partition:
                    next(b_tqdm_it)
            else:
                if action_is_fwd:
                    done_fwds += 1
                else:
                    done_bwds += 1

        # Do a scheduler step at the end of epoch if not already doing so each step.
        # Do it only when epoch is done
        if not self.trainer.PER_STEP_SCHEDULER and flush_rate < 0:
            self.lr_scheduler.step()
        futures_handler.clean_train()


#################
# GPIPE
#################
class GPipePartitionManager(SinglePartitionManager):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.saved_for_backward = dict()

    def _init_partition(self, partition, use_recomputation, disable_input_clone, req_grad):
        # NOTE: it will be called from super().__init__
        TO_DEVICE = False
        is_last_partition = self.is_last_partition
        is_first_partition = self.is_first_partition
        device = self.device

        # Set partition.
        if use_recomputation:
            if is_last_partition:
                partition_cls = GPipeLastPartition
            elif is_first_partition:
                partition_cls = GPipeFirstPartition
            else:
                partition_cls = GPipePartition

            if disable_input_clone:
                # We do the clone ourself.
                partition_cls._CLONE_INPUTS = False
            self.partition = partition_cls(partition,
                                           device,
                                           to_device=TO_DEVICE,
                                           req_grad=req_grad,
                                           )
        else:
            # Partition without recomputation
            raise NotImplementedError("GPIPE stages without recomputation not yet supported")

        if not TO_DEVICE:
            self.partition.to(device)

    def run_batch_forward(self, batch_idx, num_batches, done_bwds=None):
        """ Handles the forward pass, for last partition also prepares for the backward pass.
            by saving last result

            Algorithm:
                - Get the data
                - Forward pass
                - Send to next partition
                - for last micro batch - behave differently.

            In more detail:
                # (1) PRELOAD
                # (2) the actual forward
                # (3) send activation (if not last partition)

            Feature:
                - Pre load Y to last partition if possible
        """
        partition = self.partition
        is_training = partition.training
        last_due_end = batch_idx + 1 == num_batches
        last_due_step_every = ((batch_idx + 1) % self.step_every) == 0
        is_last_micro_batch = last_due_step_every or last_due_end

        partition.is_last_micro_batch = is_last_micro_batch

        preload_input_partition, preload_input_to_outside_loss = self.data_propagator.preload_from_dataloader(
            self.dl_iter)

        request_objects, x, ctx = self.forward_pass_and_send(
            batch_idx, num_batches, preload_input_partition)

        if not self.is_last_partition:
            return request_objects

        else:
            # Last partition - also do backward.
            ctx = (*preload_input_to_outside_loss, *ctx)
            if not is_training:
                # In Eval: Just calculate statistics.
                self.trainer.calc_test_stats(x, *ctx)
                return []
            elif is_last_micro_batch:
                # Save the out for later, when we don't do recomputation

                # NOTE: for the micro batch (no recomputation), we have x as root of the computation graph.
                # otherwise, it can be saved just for statistics, and we need to do recomputation.

                # NOTE: when we do recomputation -  this is not needed.
                # but we can use this to assert recomputation is correct.

                self.saved_for_backward[batch_idx] = (x, *ctx)
            else:
                self.saved_for_backward[batch_idx] = ctx

    def last_partition_batch_backward(self, batch_idx: int, num_batches: int, next_backward_batch_idx=None):
        # NOTE: Partition already knows if its the last micro batch, from backward
        ##############################
        # Last partition backward
        ##############################

        last_due_step_every = ((batch_idx + 1) % self.step_every) == 0
        last_due_end = batch_idx + 1 == num_batches
        is_last_micro_batch = last_due_step_every or last_due_end
        partition = self.partition
        trainer = self.trainer
        partition.is_last_micro_batch = is_last_micro_batch

        # we actually step and change LR at the FIRST micro batch
        is_first_micro_batch = (batch_idx % self.step_every) == 0
        do_step = is_first_micro_batch
        is_final_shorter_batch = (batch_idx + self.step_every) > num_batches
        change_lr = do_step and is_final_shorter_batch

        # Get the root of the computation graph.
        # NOTE: we assume did_recomputation = is_last_micro_batch
        # can also do here explicit check if we grad_fn is not NONE
        # To support no-recomputation at the last partition.
        if not is_last_micro_batch:
            self.partition.recompute(batch_idx)
            # HACK see pop_saved_graph_head.
            x = self.partition.pop_saved_graph_head(batch_idx)
            if not isinstance(x, Tensor):
                assert (len(x) == 1)
                x = x[0]
            ctx = self.saved_for_backward.pop(batch_idx)
        else:
            (x, *ctx) = self.saved_for_backward.pop(batch_idx)

        # Backprop
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)
        else:
            step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)

        # Send partition border gradients
        request_objects = self.comm_handler.send_gradients(
            partition.get_grad(batch_idx), batch_idx)

        if change_lr:
            # Scale down the learning rate, and then restore.
            old_lrs = self.scale_lr(self.reminder_scaler_lr_factor)
        else:
            old_lrs = None

        # Step
        trainer.last_partition_step_and_statistics(x,
                                                   *ctx,
                                                   step_and_stats_ctx,
                                                   step=do_step,
                                                   old_lrs=old_lrs)

        # Print training statistics.
        self.maybe_log_lr()
        return request_objects

    def run_batch_backward(self, batch_idx: int, num_batches: int, next_backward_batch_idx=None):
        """ Runs the backwards pass + step for all partitions except the last partition """

        last_due_step_every = ((batch_idx + 1) % self.step_every) == 0
        last_due_end = batch_idx + 1 == num_batches
        self.comm_handler.pre_recv_gradients(batch_idx, num_batches, last_due_end)

        is_last_micro_batch = last_due_step_every or last_due_end
        partition = self.partition
        partition.is_last_micro_batch = is_last_micro_batch  # No recomputation needed

        # Gradient aggregation:
        # we actually step at the FIRST micro batch
        is_first_micro_batch = (batch_idx % self.step_every) == 0
        do_step = is_first_micro_batch

        # Order is important
        if not is_last_micro_batch:
            self.partition.recompute(batch_idx)
        g = self.comm_handler.wait_recv_gradients(batch_idx, last_due_end)

        # recv the next gradients
        if next_backward_batch_idx is not None:
            next_backward_batch_idx_last_due_end = next_backward_batch_idx + 1 == num_batches
            self.comm_handler.pre_recv_gradients(next_backward_batch_idx, num_batches,
                                                 next_backward_batch_idx_last_due_end)

        # Compute gradients
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                partition.backward_from_recomputed(g, batch_idx)
        else:
            partition.backward_from_recomputed(g, batch_idx)

        # recompute and send backward
        request_objects = None
        if not (self.is_first_partition):
            g = partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)
        del g

        if do_step:
            trainer = self.trainer

            is_final_shorter_batch = (batch_idx + self.step_every > num_batches)
            # Sometimes last batch is smaller and needs smaller LR.
            if is_final_shorter_batch:
                old_lrs = self.scale_lr(self.reminder_scaler_lr_factor)
            else:
                old_lrs = None

            # Do the actual step.
            trainer.non_last_partition_step(old_lrs)

        return request_objects

    def run_until_flush(self, num_batches, flush_rate=-1):
        """
        Requires:
            set_dataloader() was called (in case stage requires data loading)
            train() was called

        # NOTE: its different from async pipeline
        """
        done_bwds = 0
        done_fwds = 0

        reminder = num_batches % self.step_every
        # HACK: batch_idx always less than num batches
        self.first_effected_batch = num_batches - reminder
        self.reminder_scaler_lr_factor = reminder / self.step_every

        stage_depth = self.stage_depth
        pipeline_depth = self.pipeline_depth

        work_scheduler = self.work_scheduler
        is_last_partition = self.is_last_partition
        run_batch_backward = self.run_batch_backward if not is_last_partition else self.last_partition_batch_backward
        run_batch_forward = self.run_batch_forward
        futures_handler = self.futures_handler

        mark_bwd_start = 0  # To handle LIFO

        if is_last_partition:
            b_tqdm = tqdm(range(num_batches), desc="Train")
            b_tqdm_it = iter(b_tqdm)

        while done_bwds < num_batches:
            # Act according to some policy
            action_is_fwd = work_scheduler(stage_depth, pipeline_depth, num_batches,
                                           done_fwds, done_bwds)
            if action_is_fwd:
                # micro_batch_to_run = done_fwds - done_bwds
                ro = run_batch_forward(done_fwds,
                                       num_batches,
                                       done_bwds=done_bwds)

                futures_handler.after_forward(ro, done_fwds, True)

                done_fwds += 1
            else:
                # NOTE: we want LIFO order
                if done_fwds == done_bwds + self.step_every or done_bwds == self.first_effected_batch:
                    mark_bwd_start = done_bwds

                micro_batch_to_run = done_fwds - 1 - done_bwds
                batch_idx_to_run = mark_bwd_start + micro_batch_to_run

                if not is_last_partition and done_bwds + 1 < done_fwds:
                    next_backward_micro_batch_idx = done_fwds - 1 - (done_bwds + 1)
                    next_backward_batch_idx_to_run = mark_bwd_start + next_backward_micro_batch_idx
                else:
                    # TODO: can catch the first bwd micro-batch of the next mini-batch
                    next_backward_batch_idx_to_run = None

                ro = run_batch_backward(batch_idx_to_run, num_batches,
                                        next_backward_batch_idx=next_backward_batch_idx_to_run)
                futures_handler.after_backward(ro, done_bwds)

                done_bwds += 1

                if is_last_partition:
                    # NOTE: its more accurate to measure by first partition, actually.
                    next(b_tqdm_it)

        # Do a scheduler step at the end of epoch if not already doing so each step.
        if not self.trainer.PER_STEP_SCHEDULER and flush_rate < 0:
            self.lr_scheduler.step()
        futures_handler.clean_train()
