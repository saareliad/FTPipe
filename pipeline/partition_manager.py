import torch
from torch import Tensor
import logging
from collections import OrderedDict
from . import CommunicationHandlerBase
from .partition import (Partition, LastPartition, FirstPartition,
                        PartitionWithoutRecomputation,
                        LastPartitionWithLabelInput)

from .partition import (GPipePartition, GPipeFirstPartition,
                        GPipeLastPartition, GPipeLastPartitionWithLabelInput)

from .partition import get_buffers_for_ddp_sync
from .training.interface import PartitionedTrainer  # , GradNormStepper
from .tasks import DLTask
from .weight_prediction.interface import WeightPredictor
from .gap_aware import GapAwareBase
from .work_schedulers import WorkScheduler, get_fwds_between_first_and_seconds_step_for_stage
from .weight_stashing import WeightStasher
import numpy as np
import types
from .true_weights_storage import TrueWeightsStorage

DEBUG_FAKE_DRAW = False


class SinglePartitionManager:
    PROBLEMATIC_POLICY = 'SAME'

    def __init__(self,
                 stage,
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
                 use_pre_loaded_label_input=False,
                 weight_stashing_just_for_stats=False,
                 stateless_tied=False,
                 ):
        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.
        if stateless_tied and (is_first_partition or is_last_partition):
            self.sent_obejct_patience = num_stages - 2
        else:
            self.sent_obejct_patience = 1

        # Preloaded input for last partition
        self.use_pre_loaded_label_input = use_pre_loaded_label_input

        if (gap_aware_just_loss and (not use_recomputation)):
            raise NotImplementedError(
                "gap_aware_just_loss works only with recomputation on")

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

        # NOTE: num batches just have to be high enough for the calculation.
        fwds, is_problematic = get_fwds_between_first_and_seconds_step_for_stage(
            self.work_scheduler, self.stage, self.num_stages, num_batches=390)
        self.is_problematic = is_problematic
        if is_problematic:
            print(
                f"-V- Patching problematic batches {fwds} for stage {self.stage}"
            )

        # Initialize buffers
        self.comm_handler.init_buffers()

        self.weight_predictor = None
        self.gap_aware = None
        self.weight_stasher = None

        # State for train logging
        self.log_frequency = log_frequency
        self.batches = 0

        # State for saving current relevant weight.
        self.true_weights_storage = None

        # Holds Async handle objects (for isends)
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

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
        se = self.step_every
        do_step = (batch_idx % se) == (se - 1)
        return do_step

    def set_task(self, task: DLTask):
        self.task = task

    def set_trainer(self, trainer: PartitionedTrainer):
        self.trainer = trainer

    def set_dataloader(self,
                       dataloader,
                       run_limit=-1,
                       fake_draw=DEBUG_FAKE_DRAW):
        assert self.is_first_partition or self.is_last_partition
        # self.dataloader = dataloader
        self.dl_iter = iter(dataloader)

        if fake_draw and (run_limit < len(dataloader)):
            fake_draw = len(dataloader) - run_limit
            for _ in range(fake_draw):
                next(self.dl_iter)

    def set_weight_predictor(self, weight_predictor: WeightPredictor,
                             nag_with_predictor: bool):
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

        if self.is_problematic:
            if self.PROBLEMATIC_POLICY == 'SAME':
                if self.weight_predictor is None:
                    weight_stasher.set_problematic(forward=True,
                                                   policy='CHANGE')
                else:
                    weight_stasher.set_problematic(forward=True,
                                                   policy='EVERY_BATCH')

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
            Also handles buffer sync in case stage is replicated
        """
        self.comm_handler.eval()
        self.partition.eval()
        if self.is_replicated and self.sync_buffers:
            self.comm_handler.sync_buffers(self.buffers_to_sync)

    def wait_on_sent_object(self, is_fwd, fin=False, clean_first=True):
        # TODO: can write the entire thing MUCH more nicely
        # if we just save asside and insert the new objects at the end.

        obj_holder = self.async_fwd_objects if is_fwd else self.async_bwd_objects
        # Attempt to clean all done object for saving memory
        # NOTE: this should be removed when this is supported by pytorch.
        if clean_first:
            to_del = []
            for i in obj_holder:
                a, b = obj_holder[i]
                to_remove = [i for i, r in enumerate(a) if r.is_completed()]
                for x in sorted(to_remove, reverse=True):
                    del a[x]
                # break early for simplicity
                if not a:
                    to_del.append(i)
                else:
                    break

            for i in sorted(to_del, reverse=True):
                del obj_holder[i]

            if not obj_holder:
                return

        if not fin and (len(obj_holder) <= self.sent_obejct_patience):
            return

        # Pop the item that was increased first.
        _, (sent_request_objects,
            tmp_sent_items) = obj_holder.popitem(last=False)
        for i in sent_request_objects:
            i.wait()
            # NOTE: we remove the wait() for easier debugging: can pause the debugger and find deadlocks
            # TODO: we wait here for something sent for proc 4. with `patience` of 1
            # so need to increace patience somehow.
            # while (not i.is_completed()):
            #     pass

    def get_input_data_forward(self, batch_idx, num_batches):
        """ Get the data to do forward on """
        if self.is_first_partition:
            # this also handle getting y with separate dataloader.
            x, *ctx = self.task.unpack_data_for_partition(next(self.dl_iter))
            return x, ctx
        x = self.comm_handler.get_data_forward(batch_idx, num_batches)
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
            old_lrs = None
            do_step = self.should_do_step(batch_idx)
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

        return request_objects

    def run_batch_backward(self, batch_idx, num_batches):
        """ Runs the backwards pass + step for all except the last partition """
        last_due_end = batch_idx + 1 == num_batches

        # Special case: Last batch with differnt size
        self.comm_handler.pre_recv_gradients(batch_idx, num_batches)

        weight_stasher = self.weight_stasher
        #  Recompute before waiting to the first, so parallelize communication and computation
        if weight_stasher and (not self.gap_aware_just_loss):
            # Restore to parameters which the fwd ran on
            weight_stasher.pop_and_load_stashed_params(batch_idx)

        self.partition.recompute(batch_idx)
        g = self.comm_handler.wait_recv_gradients()
        self.comm_handler.post_recv_gradients(batch_idx, num_batches)

        # Allow skiping steps (Gradient aggregation)
        old_lrs = None
        do_step = self.should_do_step(batch_idx)
        # also do step for the last. (but with smaller LR)
        if not do_step and last_due_end:
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
            g = self.partition.get_grad(batch_idx)
            request_objects = self.comm_handler.send_gradients(g, batch_idx)

        del g
        # NOTE: we can start the next recv here

        # TODO: just make it a STEP funciton
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
                    # FIXME
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

        return request_objects

    def expected_staleness(self, done_fwds, done_bwds):
        # FFFFBFBFBFBFBFBFBFBFBFBFBBBB
        # Batch | bwds   | diff | staleness
        # 0     |  0     |   0  |    0   |

        # TODO: just pre compute a table in the beggining of the run based on this.
        # I don't care too much about the formula, there is probably a nice one.
        # FIXME: for step_every > roundtrip. <----------------
        return sum(
            [self.should_do_step(x) for x in range(done_bwds, done_fwds)])

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
            sent_request_objects = run_batch_forward(done_fwds, num_batches)
            if sent_request_objects:  # last partition returns empty list.
                if async_fwd_objects:
                    wait_on_sent_object(is_fwd=True)
                async_fwd_objects[done_fwds] = sent_request_objects

        # Also clear in the end, just in case...
        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True, fin=True)

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
        is_first_partition = self.is_first_partition
        is_last_partition = self.is_last_partition
        run_batch_backward = self.run_batch_backward
        run_batch_forward = self.run_batch_forward
        async_bwd_objects = self.async_bwd_objects
        async_fwd_objects = self.async_fwd_objects
        wait_on_sent_object = self.wait_on_sent_object

        while done_bwds < num_batches:
            # Act according to some policy
            action_is_fwd = work_scheduler(stage, num_stages, num_batches,
                                           done_fwds, done_bwds)
            if action_is_fwd:
                sent_request_objects = run_batch_forward(done_fwds,
                                                         num_batches,
                                                         done_bwds=done_bwds)
                # NOTE: Last partition inserts its gradients into async_fwd_objects,
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
            wait_on_sent_object(is_fwd=True, fin=True)

        while len(async_bwd_objects) > 0:
            wait_on_sent_object(is_fwd=False, fin=True)

        # Do a scheduler step at the end of epoch if not already doing so each step.
        if not self.trainer.PER_STEP_SCHEDULER:
            self.lr_scheduler.step()


#################
# GPIPE
#################
class GPipePartitionManager(SinglePartitionManager):
    PROBLEMATIC_POLICY = "None"

    def __init__(self, *args, **kw):
        # NOTE: we changed partition type choice
        super().__init__(*args, **kw)

        self.saved_for_backward = dict()
        # assert "GPIPE" in self.work_scheduler
        # step_every

    def _init_partition(self, partition, use_recomputation):
        # NOTE: it will be called from super().__init__
        TO_DEVICE = False
        is_last_partition = self.is_last_partition
        is_first_partition = self.is_first_partition
        use_pre_loaded_label_input = self.use_pre_loaded_label_input
        device = self.device

        # Set partition.
        if use_recomputation:
            if is_last_partition:
                if use_pre_loaded_label_input:
                    partition_cls = GPipeLastPartitionWithLabelInput
                else:
                    partition_cls = GPipeLastPartition
            elif is_first_partition:
                partition_cls = GPipeFirstPartition
            else:
                partition_cls = GPipePartition
            self.partition = partition_cls(partition,
                                           device,
                                           to_device=TO_DEVICE)
        else:
            # Partition without recomputation
            # NOTE: its pretty stupied to use GPIPE in this case.
            # TODO: but I have plan doing so. "per-stage recomputation"
            raise NotImplementedError()

        if not TO_DEVICE:
            self.partition.to(device)

    def run_batch_forward(self, batch_idx, num_batches, done_bwds=None):
        """ Handles the forward pass, for last partition also handles the backward pass.

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

        is_last_micro_batch = (((batch_idx + 1) % self.step_every)
                               == 0) or batch_idx == num_batches - 1
        partition.is_last_micro_batch = is_last_micro_batch

        # Get the data to run forward on, (and target)
        preload_input = None
        if self.is_last_partition:
            preload_ctx = self.task.preload_last_partition(
                getattr(self, "dl_iter", None), self.device)
            if self.use_pre_loaded_label_input:
                preload_input = preload_ctx
                preload_ctx = tuple()

        request_objects, x, ctx = self.forward_pass_and_send(
            batch_idx, num_batches, preload_input=preload_input)

        if not self.is_last_partition:
            return request_objects

        else:
            # Last partition - also do backward.
            ctx = (*preload_ctx, *ctx)
            if not is_training:
                # In Eval: Just calculate statistics.
                self.trainer.calc_test_stats(x, *ctx)
                return []
            elif is_last_micro_batch:
                # Save the out for later, when we don't do recomputation
                # TODO: can ask trainer what exactly is neccesary from the output to save space, but its very minor.

                # NOTE: for the micro batch (no recomputation), we have x as root of the computation graph.
                # otherwise, it can be saved just for stats, and we need to do recomputation.

                # NOTE: when we do recomputation -  this is not needed.
                # but we can use this to assert recomputation is correct.

                self.saved_for_backward[batch_idx] = (x, *ctx)
            else:
                self.saved_for_backward[batch_idx] = ctx

    def last_partition_batch_backward(self, batch_idx, num_batches):
        # NOTE: Partition already knows if its the last micro batch, from backward
        ##############################
        # Last partition backward
        ##############################

        last_due_step_every = ((batch_idx + 1) % self.step_every) == 0
        last_due_end = batch_idx == (num_batches - 1)
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

        # NOTE: Usually, this is loss.backward()
        if (not do_step) and self.is_replicated:
            with self.backward_nosync_context_manager():
                step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)
        else:
            step_and_stats_ctx = trainer.backprop_last_partition(x, *ctx)

        # Send partition border gradients
        grads = partition.get_grad(batch_idx)
        request_objects = self.comm_handler.send_gradients(grads, batch_idx)

        if hasattr(trainer, "grad_norm"):
            # trainer: GradNormStepper
            trainer.grad_norm()

        if change_lr:
            # Scale down the learning rate, and then restore.
            old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)
        else:
            old_lrs = None

        # Step
        trainer.last_partition_step_and_statistics(x,
                                                   *ctx,
                                                   step_and_stats_ctx,
                                                   step=do_step)

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

        return request_objects

    def run_batch_backward(self, batch_idx, num_batches):
        """ Runs the backwards pass + step for all partitions except the last partition """

        partition = self.partition

        last_due_step_every = ((batch_idx + 1) % self.step_every) == 0
        last_due_end = batch_idx == (num_batches - 1)
        is_last_micro_batch = last_due_step_every or last_due_end
        partition.is_last_micro_batch = is_last_micro_batch

        # we actually step and change LR at the FIRST micro batch
        is_first_micro_batch = (batch_idx % self.step_every) == 0

        # Allow skiping steps (Gradient aggregation)
        do_step = is_first_micro_batch
        is_final_shorter_batch = (batch_idx + self.step_every > num_batches)
        # change_lr = do_step and is_final_shorter_batch

        self.comm_handler.pre_recv_gradients(batch_idx, num_batches)

        # TODO: consider switching order.
        if not is_last_micro_batch:
            self.partition.recompute(batch_idx)
        g = self.comm_handler.wait_recv_gradients()
        self.comm_handler.post_recv_gradients(batch_idx, num_batches)

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

            # Sometimes last batch is smaller and needs smaller LR.
            if is_final_shorter_batch:
                old_lrs, _ = self.scale_lr(self.reminder_scaler_lr_factor)
            else:
                old_lrs = None

            # if isinstance(trainer, GradNormStepper):
            if hasattr(trainer, "grad_norm"):
                # trainer: GradNormStepper
                trainer.grad_norm()

            # Do the actual step.
            trainer.non_last_partition_step()

            if old_lrs:
                # Note that sometimes its not defined locally.
                pgs = trainer.optimizer.param_groups
                for g, old_lr in zip(pgs, old_lrs):
                    g['lr'] = old_lr

        return request_objects

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called (if first partition)
            train() was called

        # NOTE: its different from async pipeline
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
        run_batch_backward = self.run_batch_backward if not is_last_partition else self.last_partition_batch_backward
        run_batch_forward = self.run_batch_forward
        async_bwd_objects = self.async_bwd_objects
        async_fwd_objects = self.async_fwd_objects
        wait_on_sent_object = self.wait_on_sent_object

        mark_bwd_start = 0  # To handle LIFO

        while done_bwds < num_batches:
            # Act according to some policy
            action_is_fwd = work_scheduler(stage, num_stages, num_batches,
                                           done_fwds, done_bwds)
            if action_is_fwd:
                # micro_batch_to_run = done_fwds - done_bwds
                sent_request_objects = run_batch_forward(done_fwds,
                                                         num_batches,
                                                         done_bwds=done_bwds)
                if sent_request_objects:
                    # wait on prev send
                    if async_fwd_objects:
                        wait_on_sent_object(is_fwd=True)
                    async_fwd_objects[done_fwds] = sent_request_objects
                done_fwds += 1
            else:
                # NOTE: we want LIFO order
                if done_fwds == done_bwds + self.step_every or done_bwds == self.first_effected_batch:
                    mark_bwd_start = done_bwds

                micro_batch_to_run = done_fwds - 1 - done_bwds
                batch_idx_to_run = mark_bwd_start + micro_batch_to_run

                sent_request_objects = run_batch_backward(
                    batch_idx_to_run, num_batches)

                if not (is_first_partition):
                    # wait on prev send
                    if async_bwd_objects:
                        wait_on_sent_object(is_fwd=False)
                    # HACK: we laizly insert at wrong index to to avoid ordering issues
                    async_bwd_objects[done_bwds] = sent_request_objects
                done_bwds += 1

        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True, fin=True)

        while len(async_bwd_objects) > 0:
            wait_on_sent_object(is_fwd=False, fin=True)

        # Do a scheduler step at the end of epoch if not already doing so each step.
        if not self.trainer.PER_STEP_SCHEDULER:
            self.lr_scheduler.step()
