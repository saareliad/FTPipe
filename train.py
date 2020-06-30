import math
import time
import torch


def training_loop(args, logger, train_dl, test_dl, is_first_partition,
                  is_last_partition, partition, statistics, train_dl_len,
                  test_dl_len, samplers):
    epochs = 0
    steps = 0
    total_epoch_times_list = []
    train_epochs_times_list = []
    # eval_epochs_times_list = []

    logger.info(f"flush rate {args.flush_rate}")
    logger.info(f"Running for {args.epochs} epochs and {args.steps} steps")
    if (args.flush_rate >= 0):
        raise NotImplementedError()

    train_batches_limit = getattr(args, "train_batches_limit", train_dl_len)
    test_batches_limit = getattr(args, "test_batches_limit", test_dl_len)

    train_batches_limit = train_dl_len if train_batches_limit < 0 else train_batches_limit
    test_batches_limit = test_dl_len if test_batches_limit < 0 else test_batches_limit

    def run_eval(eval_batches_to_run):
        logger.info(f"Running eval")
        if eval_batches_to_run == 0:
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
        # sets only to first (+last) partition
        if train_dl:
            partition.set_dataloader(train_dl, train_batches_to_run)
        # Start training
        partition.train()
        if statistics:
            statistics.train()
        # if args.flush_rate > 0:
        #     for _ in range(0, train_batches_to_run, args.flush_rate):
        #         partition.run_until_flush(
        #             min(args.flush_rate, len(train_dl)))

        #     reminder = len(train_dl) % args.flush_rate
        #     if reminder > 0:
        #         partition.run_until_flush(reminder)
        # else:
        partition.run_until_flush(train_batches_to_run)
        train_epochs_times_list.append(time.time() - train_epoch_start_time)

        if is_last_partition:
            statistics.last_partition_on_epoch_end()
        else:
            statistics.non_last_partition_on_epoch_end()
        return True

    while epochs < args.epochs or args.epochs < 0:
        for s in samplers:
            s.set_epoch(epochs)

        if args.steps > 0:
            train_batches_limit = min(train_batches_limit, args.steps - steps)

            # handle step every.
            reminder_to_drop = train_batches_limit % args.step_every
            if reminder_to_drop:
                logger.info(
                    f"Drop {reminder_to_drop} steps for each epoch>={epochs}")
                train_batches_limit -= reminder_to_drop
                if train_batches_limit <= 0:
                    break

        epoch_start_time = time.time()
        did_train = run_train(train_batches_limit)
        did_eval = run_eval(test_batches_limit)

        epochs += 1
        if did_train:
            steps += math.ceil(train_batches_limit / args.step_every)

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

        if args.steps > 0 and steps >= args.steps:
            logger.info(
                f"Finished all steps. Total steps:{steps}, rank:{args.local_rank}"
            )
            break  # steps condition met
        elif getattr(args, "patience", False):
            if is_last_partition:  # FIXME:  args.world_size - 1
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
                "early stop since valid performance hasn't improved for last {} runs"
                .format(args.patience))
            return True
        else:
            return False
