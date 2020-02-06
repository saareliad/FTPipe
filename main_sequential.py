import torch


import random
import numpy as np
import time
import logging
# TODO: fix train acc for DDP
# parse_env_vars
from main import parse_cli, parse_json_config, get_scheduler, parse_env_vars  # get_dataloaders
from models import create_normal_model_instance  # , get_partitioning
from experiments import save_experiment

# from pipeline.work_schedulers import AVAILABLE_WORK_SCHEDULERS
from optimizers import AVAILBALE_OPTIMIZERS
from pipeline.training import AVAILABLE_TRAINERS
# from pipeline.tasks import AVAILABLE_TASKS
from pipeline.stats import AVAILBALE_STATS

from pipeline.tasks import DLTask
from pipeline import dp_sim
from pipeline.partition import FirstPartition

# Distributed stuff
import torch.distributed as dist
from pipeline.util import get_world_size
from torch.nn.parallel import DistributedDataParallel
from misc.datasets import (distributed_get_train_test_dl_from_args, 
simplified_get_train_test_dl_from_args,
new_distributed_get_train_test_dl_from_args)


class SyncCVTask(DLTask):

    def __init__(self, device):
        self.device = device

    def unpack_data_for_partition(self, data):
        x, y = data
        with torch.no_grad():
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

        return x, y

    def pack_send_context(self):
        raise NotImplementedError()


class SequentailManager:
    def __init__(self, model, device, log_frequency=100, recompute=False, ddp=False, local_rank=None):

        self.device = device
        self.needs_unwrap = False
        self.logger = logging.getLogger()

        if not recompute:
            self.model = model.to(device)
            if ddp:
                self.needs_unwrap = True
                # see https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
                self.model = DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank)
                self.logger.info("-I- using DDP!")
        else:
            raise NotImplementedError()
            #  classes_list_to_patch=DEFAULT_CLASSES_LIST_TO_PATCH
            self.model = FirstPartition(model, device, to_device=True)

        # State for train logging
        self.log_frequency = log_frequency
        self.batches = 0


        # self.trainer
        # self.task

        self.task = SyncCVTask(device)


    def unwrap_model(self):
        # Have to unwrap DDP & FP16, if using.
        if self.needs_unwrap:
            return self.model.module
        else:
            return self.model

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader
        self.dl_iter = iter(self.dataloader)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called
            train() or eval() was called
        """

        for batch_idx in range(num_batches):

            data = next(self.dl_iter)

            # Unpack data
            # TODO: with task
            x, *ctx = self.task.unpack_data_for_partition(data)

            x = self.model(x)
            step_and_stats_ctx = self.trainer.backprop_last_partition(x, *ctx)
            self.trainer.last_partition_step_and_statistics(
                x, *ctx, step_and_stats_ctx)

            assert(self.model.training)
            self.batches += 1
            if self.batches % self.log_frequency == 0:
                batch_log_str = ''
                if hasattr(self.trainer, "scheduler"):
                    # Note: could be more than one LR, but we ignore this for simplicity.
                    lr = self.trainer.scheduler.get_last_lr()[0]
                    batch_log_str += '| lr {:02.4f}'.format(lr)

                self.logger.info(batch_log_str)

    def run_forward_until_flush(self, num_batches):
        # model = self.unwrap_model()
        eval_start_time = time.time()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                data = next(self.dl_iter)
                x, *ctx = self.task.unpack_data_for_partition(data)
                x = self.model(x)
                self.trainer.calc_test_stats(x,  *ctx)
        
        eval_end_time = time.time()
        # TODO: add eval time to statistics

def assert_args(args):
    assert (args.epochs >= 1 or args.steps >= 1)


def auto_file_name(args):
    assert hasattr(args, "auto_file_name")
    actual_bs_train = args.bs_train
    if args.ddp:
        actual_bs_train *= args.world_size

    s = f'{args.model}_{args.dataset}_seq_bs_{actual_bs_train}_seed_{args.seed}'
    args.out_filename = f"{args.out_filename}_{s}"
    print(f"Out File Name will be: {args.out_filename}")


def save_sequential_experiment(statistics, args):

    def carefull_del(config, name):
        if name in config:
            del config[name]

    UN_NEEDED_ARGS = ['stage', 'rank', 'local_rank', 'flush_rate', 'distributed_backend',
                      'num_chunks', 'verbose_comm', 'flush_rate', 'seed_from_cmd', 'verbose_comm', 'auto_file_name']

    if statistics:
        fit_res = statistics.get_stats()
        config = vars(args)

        # remove unneeded added args
        for name in UN_NEEDED_ARGS:
            carefull_del(config, name)

        save_experiment(args.out_filename, args.out_dir, config, fit_res)


def training_loop(args, logger, train_dl, test_dl, partition, scheduler, statistics):
    epochs = 0
    steps = 0
    logger.info(f"Running for {args.epochs} epochs and {args.steps} steps")

    if not hasattr(args, "train_batches_limit"):
        TRAIN_BATCHES_TO_RUN = len(train_dl)
    else:
        TRAIN_BATCHES_TO_RUN = getattr(args, "train_batches_limit") if getattr(
            args, "train_batches_limit") >= 0 else len(train_dl)

    if not hasattr(args, "test_batches_limit"):
        TEST_BATCHES_TO_RUN = len(test_dl)
    else:
        TEST_BATCHES_TO_RUN = getattr(args, "test_batches_limit") if getattr(
            args, "test_batches_limit") >= 0 else len(test_dl)

    while epochs < args.epochs or args.epochs < 0:
        if args.steps > 0:
            TRAIN_BATCHES_TO_RUN = min(
                TRAIN_BATCHES_TO_RUN, args.steps - steps)

        did_train = False
        did_eval = False
        epoch_start_time = time.time()

        for TRAIN in [True, False]:
            logger.info(f"Running {'train' if TRAIN else 'eval'}")

            # TRAIN_BATCHES_TO_RUN = 4
            # TEST_BATCHES_TO_RUN = 30

            if TRAIN:
                train_epoch_start_time = time.time()
                if TRAIN_BATCHES_TO_RUN == 0:
                    continue
                # Set Dataloader
                partition.set_dataloader(train_dl)
                # Start training
                partition.train()
                if statistics:
                    statistics.train()

                partition.run_until_flush(
                    min(TRAIN_BATCHES_TO_RUN, len(train_dl)))

                # TODO: support generically stepping per batch...
                scheduler.step()
                did_train = True
                steps += TRAIN_BATCHES_TO_RUN
                train_epoch_end_time = time.time()
                # TODO record it
                statistics.on_epoch_end()
            else:  # EVAL
                # Set Dataloader
                if TEST_BATCHES_TO_RUN == 0:
                    continue
                partition.set_dataloader(test_dl)
                partition.eval()

                if statistics:
                    statistics.eval()

                with torch.no_grad():  # TODO maybe remove this?
                    partition.run_forward_until_flush(
                        min(TEST_BATCHES_TO_RUN, len(test_dl)))

                did_eval = True
                statistics.on_epoch_end()

                epochs += 1

        logger.info('-' * 89)
        info_str = '| end of epoch {:3d} | time: {:5.2f}s | steps: {:5d}'.format(
            epochs, (time.time() - epoch_start_time), steps)
        if did_train:
            info_str += statistics.get_epoch_info_str(is_train=True)
        if did_eval:
            info_str += statistics.get_epoch_info_str(is_train=False)

        logger.info(info_str)
        logger.info('-' * 89)

        if args.steps > 0 and steps >= args.steps:
            logger.info(
                f"Finished all steps. Total steps:{steps}, rank:{args.local_rank}")
            break  # steps condition met


def init_DDP(args, logger):
    if args.distributed_backend == 'mpi':
        raise NotImplementedError("DDP only works with gloo and nccl backends")
    parse_env_vars(args)
    args.world_size = 4  # get_world_size(args.distributed_backend) # FIXME:
    # Initialize the distributed environment.
    dist.init_process_group(args.distributed_backend, init_method='env://')
    assert dist.get_world_size() == args.world_size
    args.rank = dist.get_rank()
    logger.info(f"Initialized process group; backend: {args.distributed_backend}, rank: {args.rank}, "
                f"local_rank: {args.local_rank}, world_size: {args.world_size}")
    

    # base_lr_batch_size = args.get("base_lr_batch_size", None)
    # if base_lr_batch_size is None:
    #     mutliplier = args.world_size
    # else:
    #     mutliplier = args.world_size *  args.bs_train  # TODO: combination with ddp sim.

    args.optimizer["args"]['lr'] *= args.world_size
    lr = args.optimizer["args"]['lr']
    logger.info(f"Applying linear scaling bs-per-worker:{args.bs_train}, lr:{lr}, workers:{args.world_size}, epochs:{args.epochs}, steps:{args.steps}")


def get_dataloaders(args):
    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    # else:
    #     dl_kw['pin_memory'] = True  # FIXME

    dl_kw['num_workers'] = args.num_data_workers
    dl_kw['drop_last'] = True

    get_dls_fn = simplified_get_train_test_dl_from_args
    if hasattr(args, 'ddp') and args.ddp:
        get_dls_fn = new_distributed_get_train_test_dl_from_args
        # num_splits=None, use_split=None
    train_dl, test_dl = get_dls_fn(
        args, verbose=False, **dl_kw)

    return train_dl, test_dl


def yuck_from_main():
    print(f"Using {torch.get_num_threads()} Threads")
    args = parse_cli()
    parse_json_config(args)
    args.is_sync = True  # Marking the experiment as sequential/syhcnronized

    # Basic Logger
    logging.basicConfig(filename='sequential.log', level=logging.DEBUG)
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    

    # Set Random Seed
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if hasattr(args, 'ddp') and args.ddp:
        init_DDP(args, logger)

    # parse_env_vars(args)
    # args.world_size = get_world_size(args.distributed_backend)

    device = torch.device('cpu' if args.cpu else f"cuda:{args.local_rank}")
    if not args.cpu:
        torch.cuda.set_device(device)

    if hasattr(args, "cudnn_benchmark") and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Create the model
    model = create_normal_model_instance(args.model)
    recompute = getattr(args, "recompute", False)
    if recompute:
        raise NotImplementedError()
        # # TODO: optionally partition the model and recompute by partitions
        # config = get_partitioning(args.model, model_instance=model)
        # print("-I- Using recomputation")

    partition = SequentailManager(
        model, device, log_frequency=100, recompute=recompute, ddp=args.ddp, local_rank=args.local_rank)

    if hasattr(args, "ddp_sim_num_gpus") and args.ddp_sim_num_gpus > 1:
        print(
            f"-I- simulating DDP accuracy with {args.ddp_sim_num_gpus} (DDP) GPUs per stage")
        dp_sim.convert_to_num_gpus(partition.model, args.ddp_sim_num_gpus)

    # model.to(device)

    assert_args(args)

    train_dl, test_dl = get_dataloaders(args)
    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer['type'])
    # task_cls = AVAILABLE_TASKS.get(args.task)  # TODO:
    optimizer_cls = AVAILBALE_OPTIMIZERS.get(args.optimizer_type)
    statistics = AVAILBALE_STATS.get(args.statistics)
    assert not (statistics is None)

    # Set optimizer
    optimizer = optimizer_cls(
        partition.model.parameters(), **args.optimizer['args'])

    # Set Scheduler
    # TODO: scheduler for sched aware prediction
    scheduler = get_scheduler(args, optimizer)
    trainer_extra_args = args.trainer['args']
    trainer = trainer_cls(model, optimizer=optimizer,
                          scheduler=scheduler, statistics=statistics, **trainer_extra_args)

    partition.set_trainer(trainer)

    if hasattr(args, "auto_file_name"):
        auto_file_name(args)

    training_loop(args, logger, train_dl, test_dl,
                  partition, scheduler, statistics)

    save_sequential_experiment(statistics, args)
    # NO_DP = True


if __name__ == "__main__":
    yuck_from_main()
