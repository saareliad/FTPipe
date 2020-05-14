import torch
import logging
from . import CommunicationHandlerBase
from .partition import (Partition, LastPartition, FirstPartition,
                        PartitionWithoutRecomputation,
                        LastPartitionWithLabelInput)

# from .partition import (GPipePartition, GPipeFirstPartition,
#                         GPipeLastPartition, GPipeLastPartitionWithLabelInput)

from .partition import get_buffers_for_ddp_sync
from .training.interface import PartitionedTrainer  # , GradNormStepper
from .tasks import DLTask
from .weight_prediction.interface import WeightPredictor
from .gap_aware import GapAwareBase
from .work_schedulers import WorkScheduler, get_fwds_between_first_and_seconds_step_for_stage
from .weight_stashing import WeightStasher
import numpy as np
from .true_weights_storage import TrueWeightsStorage

# TODO: a base class to inherit common stuff
# TODO: in this class we expect a bottleneck due to 1 buffer.

DEBUG_FAKE_DRAW = False

# NOTE:
# to send ack, USE:
# comm_handler.create_activations_recv_buffers()
# comm_handler.create_gradients_rcv_buffers()


def make_send_buff(comm_handler, shapes, dtypes, is_fwd):
    comm_handler.set_tensor_shapes(shapes)
    comm_handler.set_tensor_dtypes(dtypes)
    if is_fwd:
        comm_handler.create_activations_send_buffers()
    else:
        comm_handler.create_gradients_send_buffers()


class SinglePartitionManager:
    PROBLEMATIC_POLICY = 'SAME'

    def __init__(self,
                 stage,
                 num_stages,
                 partition: torch.nn.Module,
                 comm_handler: CommunicationHandlerBase,
                 work_scheduler: WorkScheduler,
                 training_tensor_shapes,
                 eval_tensor_shapes,
                 training_tensor_dtypes,
                 eval_tensor_dtypes,
                 device,
                 is_last_partition,
                 is_first_partition,
                 log_frequency=100,
                 max_buffers=2,
                 step_every=1,
                 keep_buffers_alive=False,
                 use_recomputation=True,
                 gap_aware_just_loss=False,
                 sync_buffers=False,
                 use_pre_loaded_label_input=False,
                 weight_stashing_just_for_stats=False,
                 stateless_tied=False,
                 last_batch_train_shapes=None,
                 last_batch_test_shapes=None):

        if (gap_aware_just_loss and (not use_recomputation)):
            raise NotImplementedError(
                "gap_aware_just_loss works only with recomputation on")

        self.sent_obejct_patience = 1
        self.use_pre_loaded_label_input = use_pre_loaded_label_input
        self.logger = logging.getLogger("msnag")  # FIXME
        self.gap_aware_just_loss = gap_aware_just_loss
        self.weight_stashing_just_for_stats = weight_stashing_just_for_stats
        self.comm_handler = comm_handler
        self.is_replicated = False
        self.sync_buffers = sync_buffers
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition
        self.stage = stage
        self.num_stages = num_stages
        self.step_every = step_every
        self.work_scheduler = work_scheduler(step_every)
        self._init_partition(partition, use_recomputation)
        if hasattr(comm_handler, "init_ddp_context"):
            ddp = comm_handler.init_ddp_context(self.partition.layers)
            self.partition.layers = ddp
            self.is_replicated = True
            self.logger.info(
                f"Initialized DDP stage replication for for stage {stage}.")
            self.backward_nosync_context_manager = ddp.no_sync
            if sync_buffers:
                self.buffers_to_sync = get_buffers_for_ddp_sync(
                    partition.layers)

        fwds, is_problematic = get_fwds_between_first_and_seconds_step_for_stage(
            self.work_scheduler, self.stage, self.num_stages, num_batches=390)
        self.is_problematic = is_problematic
        if is_problematic:
            print(
                f"-V- Patching problematic batches {fwds} for stage {self.stage}"
            )

        # Initialize buffers
        self._init_buffers(
            last_batch_test_shapes=last_batch_test_shapes,
            last_batch_train_shapes=last_batch_train_shapes,
            max_buffers=max_buffers,
            keep_buffers_alive=keep_buffers_alive,
            training_tensor_shapes=training_tensor_shapes,
            eval_tensor_shapes=eval_tensor_shapes,
            training_tensor_dtypes=training_tensor_dtypes,
            eval_tensor_dtypes=training_tensor_dtypes  # FIXME
        )

        self.weight_predictor = None
        self.gap_aware = None
        self.weight_stasher = None

        # State for train logging
        self.log_frequency = log_frequency
        self.batches = 0

        # State for saving current relevant weight.
        self.true_weights_storage = None

        self.delay_at_batch = {}

        # Hints,May be set later.
        # self.dl_iter = None
        self.task: DLTask
        self.trainer: PartitionedTrainer
        self.weight_predictor: WeightPredictor
        self.gap_aware: GapAwareBase
        self.weight_stasher: WeightStasher
        self.true_weights_storage: TrueWeightsStorage

    def _init_partition(self, partition, use_recomputation):
        TO_DEVICE = False
        is_last_partition = self.is_last_partition
        is_first_partition = self.is_first_partition
        use_pre_loaded_label_input = self.use_pre_loaded_label_input
        device = self.device

        # Set partition.
        if use_recomputation:
            if is_last_partition:
                partition_cls = LastPartition if not use_pre_loaded_label_input else LastPartitionWithLabelInput
            elif is_first_partition:
                partition_cls = FirstPartition
            else:
                partition_cls = Partition
            self.partition = partition_cls(partition,
                                           device,
                                           to_device=TO_DEVICE)
        else:
            # Partition without recomputation
            if is_last_partition:
                partition_cls = LastPartition if not use_pre_loaded_label_input else LastPartitionWithLabelInput
                self.partition = partition_cls(partition,
                                               device,
                                               to_device=TO_DEVICE)
            elif is_first_partition:
                partition_cls = PartitionWithoutRecomputation
                self.partition = partition_cls(partition,
                                               device,
                                               to_device=TO_DEVICE,
                                               _REQ_GRAD=False)
            else:
                partition_cls = PartitionWithoutRecomputation
                self.partition = partition_cls(partition,
                                               device,
                                               to_device=TO_DEVICE,
                                               _REQ_GRAD=True)
        
        # We do the clone ourself.
        # if hasattr(partition_cls, "_CLONE_INPUTS"):
        partition_cls._CLONE_INPUTS = False

        if not TO_DEVICE:
            self.partition.to(device)

    def _init_buffers(self, last_batch_test_shapes, last_batch_train_shapes,
                      max_buffers, keep_buffers_alive, training_tensor_shapes,
                      eval_tensor_shapes, training_tensor_dtypes,
                      eval_tensor_dtypes):

        # TODO

        self.last_batch_train_shapes = last_batch_train_shapes
        self.last_batch_test_shapes = last_batch_test_shapes
        self.changed_shapes_last_batch_fwd = False
        self.changed_shapes_last_batch_bwd = False
        self.max_buffers = max_buffers

        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.eval_tensor_dtypes = eval_tensor_dtypes  # FIXME

        shapes_are_equal = eval_tensor_shapes == training_tensor_shapes
        dtypes_are_equal = eval_tensor_dtypes == training_tensor_dtypes
        dtypes_and_shapes_are_equal = shapes_are_equal and dtypes_are_equal

        no_different_last_batch_shapes = (last_batch_train_shapes is
                                          None) and (last_batch_test_shapes is
                                                     None)

        if dtypes_and_shapes_are_equal and no_different_last_batch_shapes:
            # HACK: if same shapes and datatypes, the buffers can remain!
            keep_buffers_alive = True
        elif keep_buffers_alive and dtypes_and_shapes_are_equal:
            raise ValueError(
                f"got keep_buffers_alive=True, but can't because last batch has different size."
            )

        self.keep_buffers_alive = keep_buffers_alive

        self._bwd_send_buffers()

        if keep_buffers_alive:
            self._fwd_send_buffers_train()
            if not dtypes_and_shapes_are_equal:
                self.comm_handler.save_send_buffers(name="train")
                self.comm_handler.clear_send_buffers()
                self._fwd_send_buffers_eval()
                self.comm_handler.save_send_buffers(name="eval")
                self.comm_handler.use_send_buffers("train")

        self.dtypes_and_shapes_are_equal = dtypes_and_shapes_are_equal

    def _fwd_send_buffers_train(self):
        return make_send_buff(self.comm_handler, self.training_tensor_shapes,
                              self.training_tensor_dtypes, True)

    def _fwd_send_buffers_eval(self):
        return make_send_buff(self.comm_handler, self.eval_tensor_shapes,
                              self.eval_tensor_dtypes, True)

    def _bwd_send_buffers(self):
        return make_send_buff(self.comm_handler, self.training_tensor_shapes,
                              self.training_tensor_dtypes, False)

    def _ensure_bwd_send_buffers_size_set(self, last_due_end):
        # TODO: re-write, currently its inefficient
        # Special case: Last batch with differnt size
        if last_due_end and self.last_batch_train_shapes:
            # Delete previous buffers
            print(
                f"stage: {self.stage} replacing buffers for last batch, backward"
            )
            self.changed_shapes_last_batch_bwd = True
            # Create a new buffer with the new size
            make_send_buff(self.comm_handler, self.last_batch_train_shapes,
                           self.training_tensor_dtypes, False)

        elif self.changed_shapes_last_batch_bwd:
            # NOTE: this is a special case for gpipe as bwd is LIFO.
            # already change, replace:
            self.changed_shapes_last_batch_bwd = False
            self._bwd_send_buffers()

    def _ensure_fwd_send_buffers_size_set(self, last_due_end):
        # TODO: re-write, currently its inefficient
        if last_due_end and (
            (self.partition.training and self.last_batch_train_shapes) or
            (not self.partition.training and self.last_batch_test_shapes)):
            # Delete previous buffers
            print(
                f"stage: {self.stage} replacing buffers for last batch, forward"
            )
            self.changed_shapes_last_batch_fwd = True

            # Create a new buffer with the new size
            shapes = self.last_batch_train_shapes if self.partition.training else self.last_batch_test_shapes

            dtypes = self.training_tensor_dtypes if self.partition.training else self.eval_tensor_dtypes

            make_send_buff(self.comm_handler, shapes, dtypes, True)

    def set_true_weights_storage(self, true_weights_storage):
        self.true_weights_storage = true_weights_storage

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

        return old_lrs, new_lrs

    def should_do_step(self, batch_idx):
        # Returns: bool, old_lrs to restore if needed
        # TODO:
        se = self.step_every
        do_step = (batch_idx % se) == (se - 1)
        return do_step, None

    def set_task(self, task: DLTask):
        self.task = task

    def set_trainer(self, trainer: PartitionedTrainer):
        self.trainer = trainer

    def set_dataloader(self,
                       dataloader,
                       run_limit=-1,
                       fake_draw=DEBUG_FAKE_DRAW):
        assert self.is_first_partition or self.is_last_partition
        self.dl_iter = iter(dataloader)

        if fake_draw and (run_limit < len(dataloader)):
            fake_draw = len(dataloader) - run_limit
            for _ in range(fake_draw):
                next(self.dl_iter)

    def set_weight_predictor(self, weight_predictor: WeightPredictor,
                             nag_with_predictor: bool):
        self.weight_predictor = weight_predictor

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_gap_aware(self, gap_aware):
        self.gap_aware = gap_aware

    def set_weight_stasher(self, weight_stasher: WeightStasher):
        assert (weight_stasher is not None)
        if self.is_last_partition:
            raise NotImplementedError()

        if self.is_problematic:
            if self.PROBLEMATIC_POLICY == 'SAME':
                if self.weight_predictor is None:
                    weight_stasher.set_problematic(forward=True,
                                                   policy='CHANGE')
                else:
                    weight_stasher.set_problematic(forward=True,
                                                   policy='EVERY_BATCH')
        elif self.PROBLEMATIC_POLICY == 'SKIP':
            raise NotImplementedError()
            # weight_stasher.set_problematic(forward=False, policy='CHANGE')

        self.weight_stasher = weight_stasher

    def train(self):
        """Sets training mode.
            Also Handles the transition : eval -> train 
        """
        # TODO: create() should get them as parameter, instead of this set_...
        self.comm_handler.set_tensor_shapes(self.training_tensor_shapes)
        self.comm_handler.set_tensor_dtypes(self.training_tensor_dtypes)

        self.partition.train()

        if self.keep_buffers_alive and not self.dtypes_and_shapes_are_equal:
            self.comm_handler.use_send_buffers("train")
        else:
            # Forward buffers:
            # re-create if needed.
            if self.changed_shapes_last_batch_fwd:
                self.changed_shapes_last_batch_fwd = False
                self._fwd_send_buffers_train()

            # Backward buffers:
            if self.changed_shapes_last_batch_bwd:
                self.changed_shapes_last_batch_fwd = False
                self._bwd_send_buffers()  # create=True

    def eval(self):
        """Sets evaluation mode.
            Also handles the transition : train -> eval
            Also handles buffer sync in case stage is replicated
        """
        # TODO: create() should get them as parameter, instead of this set_...
        self.comm_handler.set_tensor_shapes(self.eval_tensor_shapes)
        self.comm_handler.set_tensor_dtypes(self.eval_tensor_dtypes)

        self.partition.eval()

        if self.is_replicated and self.sync_buffers:
            self.comm_handler.sync_buffers(self.buffers_to_sync)

        if self.keep_buffers_alive and not self.dtypes_and_shapes_are_equal:
            self.comm_handler.use_send_buffers("eval")
        else:
            if self.changed_shapes_last_batch_fwd:
                self.changed_shapes_last_batch_fwd = False
                self._fwd_send_buffers_eval()

    def get_input_data_forward(self, batch_idx, num_batches):
        # TODO: this entire thing should be part of the comm handler.

        # Get the data to do forward on
        if self.is_first_partition:
            # this also handle getting y with separate dataloader.
            x, *ctx = self.task.unpack_data_for_partition(next(self.dl_iter))
            return x, ctx

        # TODO:
        last_due_end = batch_idx + 1 == num_batches
        self._ensure_fwd_send_buffers_size_set(last_due_end)
        x = self.comm_handler.recv_activations(None, batch_idx)
        x = self.comm_handler.fix_after_recv(x)
        x, *ctx = self.task.unpack_data_for_partition(x)

        return x, ctx

    def forward_pass_and_send(self,
                              batch_idx,
                              num_batches,
                              preload_input=None):
        x, ctx = self.get_input_data_forward(batch_idx, num_batches)

        if (preload_input is not None) and self.is_last_partition:
            x = (*x, *preload_input)

        x = self.partition(x, batch_idx)

        request_objects = None

        # For non last partition - send forward.
        if not self.is_last_partition:
            send_ctx = self.task.pack_send_context(x, *ctx)
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

        if not self.is_first_partition:
            self.comm_handler.create_activations_recv_buffers()

        partition = self.partition
        is_training = partition.training
        if is_training:
            expected_staleness = self.expected_staleness(batch_idx, done_bwds)
            self.delay_at_batch[batch_idx] = expected_staleness

        # Get the data to run forward on, (and target)
        preload_input = None
        if self.is_last_partition:
            preload_ctx = self.task.preload_last_partition(
                getattr(self, "dl_iter", None), self.device)
            if self.use_pre_loaded_label_input:
                preload_input = preload_ctx
                preload_ctx = tuple()

        # Do the forward pass with optionals
        # optional (1): Weight Prediction
        # optional (2): Weight Stashing
        if is_training:
            weight_predictor = self.weight_predictor
            weight_stasher = self.weight_stasher
            if weight_predictor is not None:
                # TODO: last partition can do Bengio Nesterov instead of predicting.
                # Requires per partition optimizer config, or some hack.

                # NOTE: (1) we scale LR here just to tell weight predictor. will do it again when we step.
                # NOTE: (2) true_weights_storage stuff handled inside predictor.
                old_lrs = None
                if batch_idx >= self.first_effected_batch:
                    old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)

                weight_predictor.setup(expected_staleness)
                weight_predictor.forward()
                if old_lrs:
                    pgs = self.trainer.optimizer.param_groups
                    for pg, old_lr in zip(pgs, old_lrs):
                        pg['lr'] = old_lr

                request_objects, x, ctx = self.forward_pass_and_send(
                    batch_idx, num_batches, preload_input=preload_input)

                if weight_stasher is not None:
                    # Stash parameters for later.
                    # Note: wait stasher should be None be in last partition.

                    # TODO: option to do it in all except last partition.  ("NAG ONLY STALENESS 0")
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
                    batch_idx, num_batches, preload_input=preload_input)
                if weight_stasher is not None:
                    weight_stasher.stash_current(batch_idx, expected_staleness)
        else:
            # Not training. just go on as usual
            request_objects, x, ctx = self.forward_pass_and_send(
                batch_idx, num_batches, preload_input=preload_input)

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
                step_and_stats_ctx = trainer.backprop_last_partition(
                    x, *ctx)  # NOTE: Usually, this is loss

            # Send partition border gradients
            grads = partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(
                grads, batch_idx)
            # TODO: problem when sharing weights (sending nn.Parameters)
            # TODO: with weight prediction, with weight stashing.

            self.true_weights_storage.restore_if_needed()  # check=False

            if hasattr(trainer, "grad_norm"):
                # trainer: GradNormStepper
                trainer.grad_norm()

            # Step

            trainer.last_partition_step_and_statistics(x,
                                                       *ctx,
                                                       step_and_stats_ctx,
                                                       step=do_step)

            if do_step:
                self.true_weights_storage.reset_on_step()

            # Print training statistics.
            self.batches += 1
            if self.batches % self.log_frequency == 0:
                batch_log_str = ''
                if hasattr(trainer, "scheduler"):
                    # Note: could be more than one LR, but we ignore this for simplicity.
                    lr = trainer.scheduler.get_last_lr()[0]
                    batch_log_str += '| lr {:02.9f}'.format(lr)

                # TODO: add more stats. e.g can print here time, ' ms/batch {:5.2f} | ' ,...
                self.logger.info(batch_log_str)

            if old_lrs:
                # return to previous LRs.
                pgs = trainer.optimizer.param_groups
                for g, old_lr in zip(pgs, old_lrs):
                    g['lr'] = old_lr

        # request_objects.join()
        return request_objects

    def run_batch_backward(self, batch_idx, num_batches):
        """ Runs the backwards pass + step for all except the last partition """
        if not self.is_last_partition:
            self.comm_handler.create_gradients_rcv_buffers()
        # TODO:
        # # Special case: Last batch with differnt size
        last_due_end = batch_idx + 1 == num_batches
        self._ensure_bwd_send_buffers_size_set(last_due_end)

        weight_stasher = self.weight_stasher
        #  Recompute before waiting to the first, so parallelize communication and computation
        if weight_stasher and (not self.gap_aware_just_loss):
            # Restore to parameters which the fwd ran on
            weight_stasher.pop_and_load_stashed_params(batch_idx)

        self.partition.recompute(batch_idx)

        # NOTE: in MPI version there was hacky zero and sync here,,,
        g = self.comm_handler.recv_gradients(None, batch_idx)
        g = self.comm_handler.fix_after_recv(g)

        # Allow skiping steps (Gradient aggregation)
        do_step, old_lrs = self.should_do_step(batch_idx)

        # also do step for the last. (but with smaller LR)
        if not do_step and (batch_idx == (num_batches - 1)):
            do_step = True
            old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)

        # Compute gradients
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                self.partition.backward_from_recomputed(g, batch_idx)
        else:
            self.partition.backward_from_recomputed(g, batch_idx)

        # recompute and send backward
        request_objects = None
        if not (self.is_first_partition):
            # g = self.partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(
                self.partition.get_grad(batch_idx), batch_idx)

        # TODO: here we can send the next rcev buffer

        if do_step:
            trainer = self.trainer
            weight_stasher = self.weight_stasher

            # if isinstance(trainer, GradNormStepper):
            if hasattr(trainer, "grad_norm"):
                # trainer: GradNormStepper
                trainer.grad_norm()

            # TODO: allow access to real theta just for statistics
            if weight_stasher:
                if self.gap_aware_just_loss:
                    stashed_theta = weight_stasher.pop_stashed_buff(batch_idx)
                    # FIXME: the whole idea of recording the gap from here is not good.
                    pre_computed_gap = 0 if stashed_theta is None else None
                    trainer.try_record_real_gap_from_current(
                        stashed_theta, pre_computed_gap=pre_computed_gap)
                    real_theta = None
                else:
                    real_theta = self.true_weights_storage.get_true_weights()
                    # FIXME: the whole idea of recording the gap from here is not good.
                    # TODO: we can get the gap for free from gap aware sometimes.
                    trainer.try_record_real_gap_from_current(real_theta)
                    stashed_theta = None
            else:
                real_theta = None
                stashed_theta = None
            # ####### Preparing to step

            if self.gap_aware:
                # Get delay and modify gradients.
                if self.is_problematic:
                    # Average delays
                    mb = self.get_micro_batch(batch_idx)
                    delay = np.mean([
                        self.delay_at_batch.pop(batch_idx - i)
                        for i in range(0, mb + 1)
                    ])
                else:
                    delay = self.delay_at_batch.pop(batch_idx)

                # Modify gradients
                # TODO: return the gap.
                # TODO: handle grad clip here instead of in step.
                trainer.apply_gap_aware(real_theta=real_theta,
                                        delay=delay,
                                        stashed_theta=stashed_theta)

            if weight_stasher:
                # Mark previously stashed weights as dirty
                weight_stasher.mark_stashed_as_dirty()

            # Restore to previously saved parameters, so we can do the step on them.
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
            # FIXME: probably should be removed...
            if self.gap_aware_just_loss and self.weight_stasher:
                weight_stasher.pop_stashed_buff(batch_idx)

        # if not (self.is_first_partition):
        #     request_objects.join()
        return request_objects

    def expected_staleness(self, done_fwds, done_bwds):
        # FFFFBFBFBFBFBFBFBFBFBFBFBBBB
        # Batch | bwds   | diff | staleness
        # 0     |  0     |   0  |    0   |

        # TODO: just pre compute a table in the beggining of the run based on this.
        # I don't care too much about the formula, there is probably a nice one.
        # FIXME: for step_every > roundtrip. <----------------
        return sum(
            [self.should_do_step(x)[0] for x in range(done_bwds, done_fwds)])

    def run_forward_until_flush(self, num_batches):
        """
        Running evaluation (pipelined)
        Requires:
            set_dataloader() was called (if first partition)
            eval() was called
        """

        run_batch_forward = self.run_batch_forward

        for done_fwds in range(num_batches):
            ro = run_batch_forward(done_fwds, num_batches)

            # diffenret partitions behave differently..
            if isinstance(ro, list):
                for r in ro:
                    r.join()
            elif ro is not None:
                ro.join()

    def run_until_flush(self, num_batches):
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
        # is_first_partition = self.is_first_partition
        is_last_partition = self.is_last_partition
        run_batch_backward = self.run_batch_backward
        run_batch_forward = self.run_batch_forward

        while done_bwds < num_batches:
            # Act according to some policy
            action_is_fwd = work_scheduler(stage, num_stages, num_batches,
                                           done_fwds, done_bwds)
            if action_is_fwd:
                ro = run_batch_forward(done_fwds,
                                       num_batches,
                                       done_bwds=done_bwds)
            else:
                ro = run_batch_backward(done_bwds, num_batches)

            # Increase counters
            if is_last_partition:
                done_bwds += 1
                done_fwds += 1
            else:
                if action_is_fwd:
                    done_fwds += 1
                else:
                    done_bwds += 1

        # Do a scheduler step at the end of epoch if not already doing so each step.
        if not self.trainer.PER_STEP_SCHEDULER:
            self.lr_scheduler.step()
        if ro is not None:
            ro.join()
