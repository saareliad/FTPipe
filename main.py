import argparse
from pipeline import CommunicationHandler
from pipeline import SinglePartitionManager

from pipeline.training import AVAILABLE_TRAINERS
from pipeline.tasks import AVAILABLE_TASKS
from pipeline.stats import AVAILBALE_STATS
from pipeline.weight_prediction import get_sgd_weight_predictor
from pipeline.gap_aware import get_sgd_gap_aware_cls
from optimizers import AVAILBALE_OPTIMIZERS
from pipeline.util import get_world_size
import optimizers.lr_scheduler
from pipeline.work_schedulers import AVAILABLE_WORK_SCHEDULERS

import models
import numpy as np
import torch
from collections import OrderedDict
from misc.datasets import add_dataset_argument, simplified_get_train_test_dl_from_args
from misc.filelogger import FileLogger
import os
import json
from experiments import save_experiment
import time
import random


def parse_cli():
    parser = argparse.ArgumentParser(
        description='PyTorch partition as part of Async Pipeline')
    # parser.add_argument('--master_addr', default='127.0.0.1', type=str,
    #                     help="IP address of master(machine with rank 0)."
    #                     "DEPRECATED: Currently taken from env and not in use.")
    # parser.add_argument('--master_port', default=6001,
    #                     type=int, help="Port of master."
    #                     "DEPRECATED: Currently taken from env and not in use.")

    parser.add_argument('--rank', default=None,
                        type=int, help="Rank of worker")
    parser.add_argument('--local_rank', default=0,
                        type=int, help="Local rank of worker")

    # TODO: support multiple servers,
    # TODO heterogenous servers...
    # parser.add_argument('--num_ranks_in_server', default=1,
    #                     type=int, help="number of gpus per machine")

    # TODO: support mix precision, in the future
    # parser.add_argument('--fp16', action='store_true',
    #                     help='train model in fp16 precision')

    parser.add_argument('--distributed_backend',
                        choices=['gloo', 'nccl', 'mpi'], default='mpi', type=str,
                        help='distributed backend to use')

    #
    parser.add_argument('--model', choices=list(models.SUPPORTED_CONFIGS), default='wrn_16x4_p2',
                        type=str, help="name of the file with partitioning definitions")

    # Training, which are also needed for communication
    parser.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='B')

    parser.add_argument('--bs-test', type=int, help='Test batch size', default=200,
                        metavar='BT')

    # should be like `trainer` and `task` but I left it like this.
    add_dataset_argument(parser)

    parser.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=None, required=False)

    parser.add_argument('--logdir', type=str,
                        default='./logs', help="where logs and events go")

    parser.add_argument('--out-dir', '-o', type=str, help='Output folder for results',
                        default='./results', required=False)

    parser.add_argument('--data-dir', type=str,
                        help="Data directory", required=False)  # DEFAULT_DATA_DIR

    parser.add_argument('--out-filename', '-n', type=str,
                        help='Name of output file', required=False)

    parser.add_argument('--work_scheduler', type=str, help="scheduling policy to indicate when to perform forward pass",
                        choices=AVAILABLE_WORK_SCHEDULERS.keys(), default='1F1B')

    parser.add_argument('--cpu', action='store_true',
                        default=False, help="run partition on cpu")
    parser.add_argument('--num-data-workers', type=int,
                        help='Number of workers to use for dataloading', default=0)

    parser.add_argument('--trainer', help='Trainer to use',
                        choices=AVAILABLE_TRAINERS.keys(), default='dummy')
    parser.add_argument('--task', help='Task type to use',
                        choices=AVAILABLE_TASKS.keys(), default='cv')

    parser.add_argument('--statistics', help='Statistics to collect',
                        choices=AVAILBALE_STATS.keys(), default='cv')

    parser.add_argument('--optimizer_type', help='Optimizer type to use',
                        choices=AVAILABLE_TASKS.keys(), default='sgd1')

    parser.add_argument(
        "--epochs", type=int, help="Training epochs to run", default=-1, required=False)
    parser.add_argument(
        "--steps", type=int, help="Training steps to run", default=-1, required=False)
    # parser.add_argument('--linear_scaling', help="Use linear LR scaling rule", default=True)

    parser.add_argument('--debug', action='store_true', default=False,
                        help='Set to debug mpi applications. Will wait for attachment.')

    parser.add_argument('--config', help="Config File",
                        default='configs/dummy.json')

    parser.add_argument(
        "--num_chunks", help="Number of chunks for Double Buffering", type=int, default=4)

    args = parser.parse_args()

    # TODO: note, some arguments are supported only through config and not argparse.
    return args


def parse_json_config(args):
    assert(os.path.exists(args.config))

    with open(args.config, 'r') as f:
        output = json.load(f)

    # replace
    for key, value in output.items():
        # if hasattr(args, key):
        setattr(args, key, value)

    # Explicit yuck replace
    # (just to get help from argparse)
    if hasattr(output, 'optimizer'):
        if hasattr(output['optimizer'], 'type'):
            args.optimizer_type = output['optimizer']['type']

    # return output


def parse_env_vars(args):
    """
    Parses env vars (e.g from mpirun) and push them into args (overriding).
    This allows completing some "incomplete" cli-argument parsing.

    Requires:
        args = parse_cli()

    References:
        https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
    """

    if args.distributed_backend == 'mpi':
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])


def assert_args(args):
    assert (args.epochs >= 1 or args.steps >= 1)


def create_comm_handler(args, initialize_args):

    # get the parameters to create the comm handler
    comm_handler = CommunicationHandler(
        args.rank,
        args.local_rank,
        args.distributed_backend,
        args.num_stages,
        args.stage,
        *initialize_args,
        args.cpu,
        args.num_chunks,
        GRAD_UGLY_SHAMEFUL_NAME="_grad",
        verbose=args.verbose_comm if hasattr(args, "verbose_comm") else False)

    return comm_handler


def config_to_tuples_generator(configs):
    """ allows iterating with the tuple: (stage_id, inputs, outputs) """
    for i, v in configs.items():
        yield i, v['inputs'], v['outputs']


def config_to_tuples_array(configs):
    return np.array(list(config_to_tuples_generator(configs)))


# target_tensor_names = {"target", "target_length"}
# target_tensor_names = {"target"}
def tensor_tags_from_config(config, target_tensor_names, num_chunks, GRAD_UGLY_SHAMEFUL_NAME="_grad"):

    # Note: same tags for all proccess

    tensor_tags = {}
    tensor_tag = 1
    model = config_to_tuples_array(config)

    for (_, input_tensors, output_tensors) in model:
        for input_tensor in input_tensors:
            if input_tensor not in tensor_tags:
                tensor_tags[input_tensor] = tensor_tag
                tensor_tag += num_chunks
        for output_tensor in output_tensors:
            if output_tensor not in tensor_tags:
                tensor_tags[output_tensor] = tensor_tag
                tensor_tag += num_chunks
    # Create different tags for gradients
    for (_, input_tensors, output_tensors) in model:
        for input_tensor in input_tensors:
            input_tensor += GRAD_UGLY_SHAMEFUL_NAME
            if input_tensor not in tensor_tags:
                tensor_tags[input_tensor] = tensor_tag
                tensor_tag += num_chunks
        for output_tensor in output_tensors:
            output_tensor += GRAD_UGLY_SHAMEFUL_NAME
            if output_tensor not in tensor_tags:
                tensor_tags[output_tensor] = tensor_tag
                tensor_tag += num_chunks

    for target_tensor_name in sorted(target_tensor_names):
        tensor_tags[target_tensor_name] = tensor_tag
        tensor_tag += num_chunks

    # tensor_tags["ack"] = tensor_tag
    tensor_tag += num_chunks

    return tensor_tags, tensor_tag


def create_distributed_communcation_context(args, config, stage, stage_to_rank_map=None,
                                            target_tensor_names={"target"},
                                            training_tensor_dtypes={
                                                "input0": torch.int64, "target": torch.int64},
                                            training_tensor_shapes={
                                                "input0": None, "target": None},
                                            eval_tensor_shapes={
                                                "input0": None, "target": None},
                                            random_input_sample=None,
                                            bs_train=(1,),
                                            bs_test=(1,)):

    assert(len(training_tensor_shapes) == len(training_tensor_dtypes))
    # training_tensor_shapes = {"input0": input_size, "target": [args.batch_size]}
    # inputs_module_destinations={"input0": [0]} Not needed, already taken care of by config.

    # Creates and returns the following
    # target_tensor_names (input)
    # tensor_tags
    # receive_ranks
    # send_ranks
    # training_tensor_dtypes
    # training_tensor_shapes
    # eval_tensor_shapes
    # rank_in_stage
    # num_ranks_in_stage
    # ranks_in_previous_stage
    # ranks_in_next_stage

    # TODO: eval_tensor_dtypes

    #     dtypes = {"input0": torch.int64, "target": torch.int64}

    # eval_tensor_shapes = {**training_tensor_shapes}

    tensor_tags, TOTAL_TAGS = tensor_tags_from_config(
        config, target_tensor_names, args.num_chunks, GRAD_UGLY_SHAMEFUL_NAME="_grad")

    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    # TODO: support weight sharing
    # TODO: asset this is same order with Alon's code/config...
    receive_ranks = OrderedDict()
    send_ranks = OrderedDict()

    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            # Update only for this stage...
            if i != stage and j != stage:
                continue

            stage_i = config[i]
            stage_j = config[j]
            for tensor_name in stage_i['outputs']:
                if tensor_name in stage_j['inputs']:
                    if stage == j:
                        receive_ranks[tensor_name] = ranks_in_stage(i)
                    else:
                        send_ranks[tensor_name] = ranks_in_stage(j)
    # this_stage = config['stage']
    # this_stage_tensors = set(config['stage']['inputs'] + config['stage']['outputs'])

    # Run a sequential forward pass to determine:
    # training_tensor_dtypes
    # training_tensor_shapes
    # eval_tensor_shapes
    # TODO: eval_tensor_dtypes

    # with torch.no_grad():
    for i, v in config.items():
        partition = v['model']
        if i == 0:
            a = partition(random_input_sample)
        else:
            a = partition(*a)

        outputs = v['outputs']
        dtypes = tuple(j.data.dtype for j in a)

        # Concatenate shapes with expected bs_train/bs_test
        # the batch size can be a collection (e.g (batch, seq_len) in NLP)
        bs_train = to_tuple(bs_train)
        bs_test = to_tuple(bs_test)
        # TODO: this assume that batch is first
        len_bs = len(bs_train)
        train_shapes = tuple(
            tuple(list(bs_train) + list(j.data.size()[len_bs:])) for j in a)
        eval_shapes = tuple(
            tuple(list(bs_test) + list(j.data.size()[len_bs:])) for j in a)

        training_tensor_dtypes.update(zip(outputs, dtypes))
        training_tensor_shapes.update(zip(outputs, train_shapes))
        eval_tensor_shapes.update(zip(outputs, eval_shapes))

    # Create:
    # rank_in_stage
    # num_ranks_in_stage
    # ranks_in_previous_stage
    # ranks_in_next_stage

    # rank = args.local_rank
    rank_in_stage = stage_to_rank_map[stage].index(
        args.local_rank) if stage_to_rank_map else 0
    num_ranks_in_stage = len(
        stage_to_rank_map[stage]) if stage_to_rank_map else 1

    # TODO: can create these by th econfig too.
    ranks_in_previous_stage = ranks_in_stage(
        stage - 1) if stage > 0 else []
    ranks_in_next_stage = ranks_in_stage(
        stage + 1) if stage < args.num_stages - 1 else []

    comm_args = (receive_ranks,
                 send_ranks,
                 tensor_tags,
                 target_tensor_names,
                 training_tensor_dtypes,
                 rank_in_stage,
                 num_ranks_in_stage,
                 ranks_in_previous_stage,
                 ranks_in_next_stage,
                 TOTAL_TAGS)
    shapes = (training_tensor_shapes, eval_tensor_shapes)

    return comm_args, shapes


def to_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def get_scheduler(args, optimizer):
    if hasattr(args, "lr_scheduler"):
        # should_step = False
        attr = getattr(args, 'lr_scheduler')
        if attr['type'] in optimizers.lr_scheduler.AVAILABLE_LR_SCHEDULERS:
            scheduler_cls = getattr(optimizers.lr_scheduler, attr['type'])
            # should_step = True
        else:
            scheduler_cls = getattr(torch.optim.lr_scheduler, attr['type'])
        scheduler = scheduler_cls(optimizer, **attr['args'])
        # TODO: in some optimizers version we can bendfit from lr=0 (build momentum in place)
        # while on others we dont, and better step.
        # For now I just leave it as is.
        # OPTIONAL: Perform a dummy step to avoid lr=0 at the start of the training.
        # if should_step:
        #     scheduler.step()
        return scheduler


def get_gap_aware(args, optimizer):
    if not hasattr(args, 'gap_aware'):
        return None
    gap_aware_args = getattr(args, 'gap_aware')['args']
    optimizer_type = getattr(args, 'optimizer')['type']

    # if gap_aware_args['penatly_for_weight_decay']:
    #     if not (hasattr(args, "weight_predictor")):
    #         raise ValueError("Do not use penatly_for_weight_decay with msnag/DANA weight predictor")

    if not optimizer_type == 'sgd1':  # pytorch
        raise NotImplementedError()
    gap_aware_cls = get_sgd_gap_aware_cls(optimizer_type)
    return gap_aware_cls(optimizer, **gap_aware_args)


def get_weight_predictor(args, optimizer, scheduler=None):
    """
        Returns:
            weight_predictor,
            nag_with_predictor: bool
    """
    if not hasattr(args, 'weight_prediction'):
        return None, None

    optimizer_type = getattr(args, 'optimizer')['type']
    pred = getattr(args, 'weight_prediction')
    pred_mem = pred['args']['pred_mem']
    nag_with_predictor = pred['args']['nag_with_predictor']

    assert(pred_mem in {"clone", "calc"})
    assert(pred['type'] == "msnag")
    assert('sgd' in optimizer_type)

    # if pred_mem['type'] == "msnag":
    if 'sgd' in optimizer_type:
        weight_predictor = get_sgd_weight_predictor(
            optimizer_type, pred_mem, optimizer, scheduler)
        return weight_predictor, nag_with_predictor
    else:
        raise NotImplementedError()


def hack_trainer_to_gap_aware(args):
    if hasattr(args, 'gap_aware'):
        # on = args.gap_aware['on']
        assert args.gap_aware['policy'] == 'almost_last_partition'

        is_almost_last_partition = args.local_rank == args.world_size - 2

        # HACK: change trainer name
        if is_almost_last_partition:
            args.trainer += "_gap_aware"
            return True

    return False


def main():
    args = parse_cli()
    parse_json_config(args)
    parse_env_vars(args)
    use_gap_aware = hack_trainer_to_gap_aware(args)

    if args.debug:
        import ptvsd
        port = 3000 + args.local_rank
        ptvsd.enable_attach(address=('127.0.0.1', port))
        ptvsd.wait_for_attach()

    args.world_size = get_world_size(args.distributed_backend)
    # is_last_rank = args.local_rank == get_world_size(args.distributed_backend) - 1

    logger = FileLogger(args.logdir, global_rank=args.rank,
                        local_rank=args.local_rank, name='msnag', world_size=args.world_size)

    assert_args(args)
    configs = models.get_partitioning(args.model, model_instance=None)

    input_names = configs.pop('model inputs')
    output_names = configs.pop('model outputs')

    if not args.seed:
        args.seed = random.randint(0, 2 ** 31)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    NO_DP = True
    # TODO: make it nicer.
    stage = None
    if NO_DP:
        args.num_stages = len(configs)
        stage = args.local_rank
        is_first_partition = args.local_rank == 0
        is_last_partition = args.local_rank == len(configs) - 1
        args.num_ranks = len(configs)
    else:
        raise NotImplementedError()

    assert(not (stage is None))
    args.stage = stage

    if use_gap_aware:
        logger.info(f"Stage {args.stage} will use Gap Aware")

    # Here is a dummy for For CIFAR10 network
    # TODO: best practice is env var for choosing gpu
    device = torch.device('cpu' if args.cpu else f"cuda:{args.local_rank}")
    if not args.cpu:
        torch.cuda.set_device(device)

    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    # else:
    #     dl_kw['pin_memory'] = True  # FIXME

    dl_kw['num_workers'] = args.num_data_workers
    dl_kw['drop_last'] = True

    train_dl, test_dl = simplified_get_train_test_dl_from_args(
        args, verbose=False, **dl_kw)

    # TODO: do the following block generically and automatically using tasks.
    # May need to extend tasks a little to produce all targets, etc.
    # Will be easier when we work we precise example from NLP which is different from CV.
    x, y = next(iter(train_dl))
    BASE_INPUT_SHAPE = x.shape[1:]
    BASE_TARGET_SHAPE = y.shape[1:]

    bs_train = to_tuple(args.bs_train)
    bs_test = to_tuple(args.bs_test)
    # Smallest batch as possible.
    # we will determine the rest of the shape with bs_train, bs_test
    SAMPLE_BATCH_SIZE = 1
    random_input_sample = torch.randn(SAMPLE_BATCH_SIZE, *BASE_INPUT_SHAPE)

    # TODO formalize with function according to dataset/task
    target_tensor_names = {"target"}
    # training_tensor_dtypes = {"input0": torch.int64, "target": torch.int64}
    training_tensor_dtypes = {"input0": x.dtype, "target": y.dtype}

    training_tensor_shapes = {"input0": (
        *bs_train, *BASE_INPUT_SHAPE), "target": (*bs_train, *BASE_TARGET_SHAPE)}

    eval_tensor_shapes = {"input0": (
        *bs_test, *BASE_INPUT_SHAPE), "target": (*bs_test, *BASE_TARGET_SHAPE)}

    comm_init_args, shapes = \
        create_distributed_communcation_context(args, configs, stage,
                                                stage_to_rank_map=None,
                                                target_tensor_names=target_tensor_names,
                                                training_tensor_dtypes=training_tensor_dtypes,
                                                training_tensor_shapes=training_tensor_shapes,
                                                eval_tensor_shapes=eval_tensor_shapes,
                                                random_input_sample=random_input_sample,
                                                bs_train=bs_train,
                                                bs_test=bs_test)

    comm_handler = create_comm_handler(args, comm_init_args)
    # init_dist(args)

    training_tensor_shapes, eval_tensor_shapes = shapes

    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer)
    task_cls = AVAILABLE_TASKS.get(args.task)
    optimizer_cls = AVAILBALE_OPTIMIZERS.get(args.optimizer_type)
    statistics = AVAILBALE_STATS.get(args.statistics)
    work_scheduler = AVAILABLE_WORK_SCHEDULERS.get(args.work_scheduler)

    partition = SinglePartitionManager(
        stage,
        configs, configs[stage]['model'],
        comm_handler, work_scheduler(), training_tensor_shapes,
        eval_tensor_shapes,
        device, is_last_partition, is_first_partition)

    # Set Trainer
    # FIXME: this `if` is just to support trainers hardcoded with their own default optimizer.
    # currently thats just the `dummy` trainer,
    # This is to be removed

    optimizer = optimizer_cls(
        partition.partition.parameters(), **args.optimizer['args'])

    scheduler = get_scheduler(args, optimizer)

    # TODO: scheduler for sched aware prediction
    weight_predictor, nag_with_predictor = get_weight_predictor(
        args, optimizer, scheduler=scheduler)

    # trainer_args = dict(optimizer=optimizer,
    #                     scheduler=scheduler, statistics=statistics)

    # is_almost_last_partition = args.local_rank == args.world_size - 2

    if hasattr(trainer_cls, "HAS_GAP_AWARE"):
        gap_aware = get_gap_aware(args, optimizer)
        # trainer_args['gap_aware'] = gap_aware
        trainer = trainer_cls(gap_aware, partition.partition, optimizer=optimizer,
                              scheduler=scheduler, statistics=statistics)
    else:
        trainer = trainer_cls(partition.partition, optimizer=optimizer,
                              scheduler=scheduler, statistics=statistics)

    partition.set_trainer(trainer)
    if weight_predictor:
        partition.set_weight_predictor(weight_predictor, nag_with_predictor)

    # Set Task
    task = task_cls(device, is_last_partition, is_first_partition)
    partition.set_task(task)

    epochs = 0
    steps = 0
    logger.info(f"flush rate {args.flush_rate}")
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
        did_train = False
        did_eval = False
        epoch_start_time = time.time()
        # steps_at_epoch_start = steps
        for TRAIN in [True, False]:
            logger.info(f"Running {'train' if TRAIN else 'eval'}")

            # TRAIN_BATCHES_TO_RUN = 4
            # TEST_BATCHES_TO_RUN = 30

            if TRAIN:
                # Set Dataloader
                # sets only to first partition
                partition.set_dataloader(train_dl)
                # Start training
                partition.train()
                if statistics:
                    statistics.train()
                if args.flush_rate > 0:
                    for _ in range(0, TRAIN_BATCHES_TO_RUN, args.flush_rate):
                        if use_gap_aware:
                            # Gap calculation assumes staleness 1, so skipping first batch in flush.
                            gap_aware.skip_one_apply()
                        partition.run_until_flush(
                            min(args.flush_rate, len(train_dl)))

                    reminder = len(train_dl) % args.flush_rate
                    if reminder > 0:
                        if use_gap_aware:
                            gap_aware.skip_one_apply()
                        partition.run_until_flush(reminder)
                else:
                    partition.run_until_flush(
                        min(TRAIN_BATCHES_TO_RUN, len(train_dl)))

                scheduler.step()
                if use_gap_aware:
                    gap_aware.update_max_lr()
                did_train = True
            else:
                # Set Dataloader
                # sets only to first partition
                if TEST_BATCHES_TO_RUN == 0:
                    continue
                partition.set_dataloader(test_dl)
                partition.eval()

                if statistics:
                    statistics.eval()
                    with torch.no_grad():
                        partition.run_forward_until_flush(
                            min(TEST_BATCHES_TO_RUN, len(test_dl)))

                did_eval = True
            if args.local_rank == args.world_size - 1:
                statistics.on_epoch_end()
        
        epochs += 1
        if did_train:
            steps += TRAIN_BATCHES_TO_RUN

        if args.local_rank == args.world_size - 1:
            logger.info('-' * 89)
            # ms/batch {:5.2f}
            info_str = '| end of epoch {:3d} | time: {:5.2f}s | steps: {:5d}'.format(
                epochs, (time.time() - epoch_start_time), steps)
            if did_train:
                info_str += statistics.get_epoch_info_str(is_train=True)
            if did_eval:
                info_str += statistics.get_epoch_info_str(is_train=False)

            logger.info(info_str)
            logger.info('-' * 89)

        if args.steps > 0 and steps >= args.steps:
            break  # steps condition met

    # FIXME: If last state
    if args.rank == get_world_size(args.distributed_backend) - 1:
        if statistics:
            fit_res = statistics.get_stats()
            config = vars(args)  # FIXME: TODO:
            save_experiment(args.out_filename, args.out_dir, config, fit_res)
    torch.distributed.barrier()

    # TODO:


if __name__ == "__main__":
    main()
