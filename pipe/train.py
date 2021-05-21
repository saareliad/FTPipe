import math
import os
import time
import warnings

import torch

from pipe.pipeline import SinglePartitionManager
from pipe.pipeline.trainers.statistics import Stats
from pipe.prepare_pipeline import SmallerLastBatchPolicy, DEFAULT_STEP_EVERY_SMALLER_LAST_BATCH_POLICY


def training_loop(args, logger, train_dl, test_dl,
                  is_last_partition, partition: SinglePartitionManager, statistics: Stats, train_dl_len,
                  test_dl_len, samplers):
    # Prepare for loop
    last_batch_smaller_n_micro_batches_policy = getattr(args, "last_batch_smaller_n_micro_batches_policy",
                                                        DEFAULT_STEP_EVERY_SMALLER_LAST_BATCH_POLICY)

    save_checkpoint_every_x_epochs = approximate_checkpoint_every_x_epochs(args, train_dl_len)

    epochs = 0
    steps = 0
    total_epoch_times_list = []
    train_epochs_times_list = []
    cp_saver = CheckpointsSaver(args)

    logger.info(f"flush rate {args.flush_rate}")
    logger.info(f"Running for {args.epochs} epochs and {args.steps} steps")
    if args.flush_rate >= 0:
        raise NotImplementedError()

    train_batches_limit = getattr(args, "train_batches_limit", train_dl_len)
    test_batches_limit = getattr(args, "test_batches_limit", test_dl_len)

    if getattr(args, "train_batches_limit", -1) > 0:
        warnings.warn(
            "(dev feature) hard limiting train batches per flush: "
            "different last batch not supported, messages may get truncated")

    if getattr(args, "test_batches_limit", -1) > 0:
        warnings.warn(
            "(dev feature) hard limiting test batches per flush: "
            "different last batch not supported, messages may get truncated")

    train_batches_limit = train_dl_len if train_batches_limit < 0 else train_batches_limit
    test_batches_limit = test_dl_len if test_batches_limit < 0 else test_batches_limit

    # Here comes utility functions:     run_eval and run_train
    def run_eval(eval_batches_to_run):
        logger.info(f"Running eval")
        if eval_batches_to_run == 0:
            partition.eval()
            if statistics:
                statistics.eval()
            return False
        if test_dl:
            partition.set_dataloader(test_dl, eval_batches_to_run)
        partition.eval()
        if statistics:
            statistics.eval()

        with torch.no_grad():  # TODO maybe remove this?
            partition.run_forward_until_flush(eval_batches_to_run)

        # eval_epochs_times_list.append(time.time() - eval_epoch_start_time)
        if is_last_partition:
            statistics.last_partition_on_epoch_end()
        # NOTE: in eval() only last partition computes statistics
        # else:
        #     statistics.non_last_partition_on_epoch_end()
        return True

    def run_train(train_batches_to_run):
        logger.info(f"Running train")

        train_epoch_start_time = time.time()
        if train_batches_to_run == 0:
            return False
        # Set Dataloader
        if train_dl:
            partition.set_dataloader(train_dl, train_batches_to_run)
        # Start training
        partition.train()
        if statistics:
            statistics.train()

        if args.flush_rate > 0:
            for _ in range(0, train_batches_to_run, args.flush_rate):
                partition.run_until_flush(args.flush_rate)
            reminder = train_batches_to_run % args.flush_rate
            if reminder > 0:
                logger.info(f"Warning: will run for reminder {reminder} to finish epoch")
                partition.run_until_flush(reminder)
                # TODO: allow statistics between flushes (e.g eval)

            if not partition.trainer.PER_STEP_SCHEDULER:
                partition.lr_scheduler.step()
        else:
            partition.run_until_flush(train_batches_to_run)

        train_epochs_times_list.append(time.time() - train_epoch_start_time)

        if is_last_partition:
            statistics.last_partition_on_epoch_end()
        else:
            statistics.non_last_partition_on_epoch_end()
        return True

    # Actual training loop
    while epochs < args.epochs or args.epochs < 0:
        for s in samplers:
            s.set_epoch(epochs)

        (reminder_micro_batches,
         train_batches_limit_to_use) = get_micro_batches_until_flush(args,
                                                                     train_batches_limit,
                                                                     steps,
                                                                     last_batch_smaller_n_micro_batches_policy,
                                                                     logger, partition)

        if train_batches_limit_to_use <= 0:
            logger.info(
                f"breaking early: "
                f" can't complete a full step with {args.step_every} gradient accumulations.")
            break
        epoch_start_time = time.time()

        # TODO: flush every 1000
        did_train = run_train(train_batches_limit_to_use)

        did_eval = run_eval(test_batches_limit)

        epochs += 1
        if did_train:
            floor_steps = args.steps > 0 \
                          and reminder_micro_batches \
                          and last_batch_smaller_n_micro_batches_policy == SmallerLastBatchPolicy.DropReminder
            if floor_steps:
                steps += math.floor(train_batches_limit_to_use / args.step_every)
            else:
                steps += math.ceil(train_batches_limit_to_use / args.step_every)

        is_last = (0 < args.epochs <= epochs) or (0 < args.steps <= steps)
        if is_last or epochs % save_checkpoint_every_x_epochs == 0:
            cp_saver.maybe_save_checkpoint(partition.partition.layers, steps)

        total_epoch_time = (time.time() - epoch_start_time)
        total_epoch_times_list.append(total_epoch_time)
        # if is_last_partition
        if args.local_rank == args.world_size - 1:
            logger.info('-' * 89)
            # ms/batch {:5.2f}
            info_str = '| end of epoch {:3d} | time: {:5.2f}s | steps: {:5d}'.format(
                epochs, total_epoch_time, steps)
            if did_train:
                info_str += statistics.get_epoch_info_str(is_train=True)
            if did_eval:
                info_str += statistics.get_epoch_info_str(is_train=False)

            logger.info(info_str)
            logger.info('-' * 89)

        if 0 < args.steps <= steps:
            logger.info(
                f"Finished all steps. Total steps:{steps}, rank:{args.local_rank}"
            )
            break  # steps condition met
        elif getattr(args, "patience", False):
            if args.world_size - 1:
                assert is_last_partition
                # TODO: Try catch?                    
                should_early_stop = should_stop_early(
                    args, statistics.get_metric_for_early_stop(), logger)
                data = torch.tensor(int(should_early_stop))
            else:
                data = torch.tensor(int(False))  # create buffer

            torch.distributed.broadcast(data, args.world_size - 1)
            should_early_stop = data.item()
            if should_early_stop:
                break

    return total_epoch_times_list, train_epochs_times_list


def get_micro_batches_until_flush(args, train_batches_limit, steps, step_every_smaller_last_batch_policy,
                                  logger, partition):
    if args.steps > 0:
        steps_left = args.steps - steps
        # TODO: it can be more fine-grained depends on policy but I leave it for now.
        batches_left = steps_left * args.step_every
        train_batches_limit_to_use = min(train_batches_limit, batches_left)

        if batches_left < train_batches_limit:
            # Re-define last batch train shapes.
            # now, the last batch shapes are not smaller.
            logger.info(
                "batches_left are smaller than dataloader or limit: killing comm_handler.last_batch_train_shapes")
            partition.comm_handler.last_batch_train_shapes = None

        # handle step every.
        # if we don't do anything, we will do:
        # `train_batches_limit` batches
        # which are (train_batches_limit // args.step_every) steps.
        # So the reminder is problematic
        # we can either:
        # (1) take a smaller step for it (proportional to number of grad accumulations taken).
        # (2) drop the reminder
        #
        # Note: (1) only effects the last batch so there is no problem with staleness.

        reminder_micro_batches = train_batches_limit_to_use % args.step_every
        if reminder_micro_batches:
            if step_every_smaller_last_batch_policy == SmallerLastBatchPolicy.DropReminder:
                # d_info = {
                #     "steps_left": steps_left,
                #     "batches_left": batches_left,
                #     "original_train_batches_limit": train_batches_limit,
                #     "train_batches_limit_until_flush": train_batches_limit_to_use,
                #     "step_every": args.step_every,
                #     "train_dl_len": train_dl_len
                # }
                logger.info(
                    f"Got reminder of {reminder_micro_batches} micro batches. Will drop them.")
                train_batches_limit_to_use -= reminder_micro_batches

            elif step_every_smaller_last_batch_policy == SmallerLastBatchPolicy.ProportionalStep:
                # TODO: to fix GPipe MPI, we can do it, but needs to be only for last batch.
                logger.info(
                    f"Got reminder of {reminder_micro_batches} micro batches. "
                    f"Will take proportional {reminder_micro_batches / args.step_every} last step")
            else:
                raise NotImplementedError(
                    f"Unknown SMALLER_LAST_BATCH_POLICY, {step_every_smaller_last_batch_policy}")
    else:
        train_batches_limit_to_use = train_batches_limit
        reminder_micro_batches = 0
    return reminder_micro_batches, train_batches_limit_to_use


def approximate_checkpoint_every_x_epochs(args, train_dl_len):
    save_checkpoint_every_x_epochs = getattr(args, "save_checkpoint_every_x_steps", None)
    approx_step_per_epoch = train_dl_len // args.step_every
    if save_checkpoint_every_x_epochs is not None:
        save_checkpoint_every_x_epochs = save_checkpoint_every_x_epochs // approx_step_per_epoch
    else:
        save_checkpoint_every_x_epochs = 1
    assert save_checkpoint_every_x_epochs >= 1
    print(f"Approximating: An epoch is approx {approx_step_per_epoch} steps.")
    print(f"Approximating: will save checkpoint every {save_checkpoint_every_x_epochs} epochs, and at the end.")
    return save_checkpoint_every_x_epochs


def should_stop_early(args, valid_loss, logger):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if getattr(args, "maximize_best_checkpoint_metric",
                                False) else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                f"early stop since valid performance hasn't improved for last {args.patience} runs")
            return True
        else:
            return False


class CheckpointsSaver:
    def __init__(self, args):
        self.args = args
        self.num_saved_checkpoints = 0

        if getattr(args, "save_checkpoints", False):
            assert hasattr(args, "checkpoints_save_dir")
            os.makedirs(args.checkpoints_save_dir, exist_ok=True)
        else:
            print("-W- will not save checkpoints")
            # (To change this, set: args.save_checkpoints=True, args.checkpoints_save_dir")

    def maybe_save_checkpoint(self, model, steps):
        args = self.args
        if not getattr(args, "save_checkpoints", False):
            return

        name_prefix = getattr(args, "checkpoints_save_name_prefix", "")
        name_prefix += f"_{self.num_saved_checkpoints}"
        # name_prefix += add_to_prefix
        fn = os.path.join(args.checkpoints_save_dir, f"{name_prefix}_Partition{args.stage}.pt")

        tik = time.time()
        torch.save(model.state_dict(), fn)
        tok = time.time()

        print(f"-V- stage {args.stage}: saving checkpoint took: {tok - tik}")
        self.num_saved_checkpoints += 1
        print(f"-I- stage {args.stage}: model checkpoint saved: {fn}")

        # Also save number of steps
        metatdata_fn = os.path.join(args.checkpoints_save_dir, f"{name_prefix}_Partition{args.stage}.steps")
        try:
            # We don't want it to kill training if it fails somehow
            with open(metatdata_fn, "w") as f:
                f.write(str(steps))
        except Exception as _:
            warnings.warn(f"Failed to save metadata for checkpoint {metatdata_fn}, ignoring exception")
