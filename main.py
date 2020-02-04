import argparse
from pipeline import CommunicationHandlerBase, get_auto_comm_handler_cls
from pipeline import SinglePartitionManager
from pipeline import BigBatchManager

from pipeline.training import AVAILABLE_TRAINERS
from pipeline.tasks import AVAILABLE_TASKS
from pipeline.stats import AVAILBALE_STATS  # , Stats
from pipeline.weight_prediction import get_sgd_weight_predictor
from pipeline.gap_aware import get_sgd_gap_aware_cls
from optimizers import AVAILBALE_OPTIMIZERS
from pipeline.util import get_world_size
import optimizers.lr_scheduler
from pipeline.work_schedulers import AVAILABLE_WORK_SCHEDULERS
from pipeline.weight_stashing import WeightStasher

import models
import numpy as np
import torch
from collections import OrderedDict
from misc.datasets import add_dataset_argument, simplified_get_train_test_dl_from_args, get_seperate_just_x_or_y_train_test_dl_from_args
from misc.filelogger import FileLogger
from pipeline import dp_sim
import os
import json
from experiments import save_experiment, load_experiment_for_update
import time
import random
import math

# TODO: support multiple servers,
# TODO heterogenous servers
# TODO: support mix precision, in the future


def parse_cli():
    # TODO: note, some arguments are supported only through config and not argparse.
    # TODO: replace all this
    # with a function to tell the avaialble options to the user,
    # as we overrride the entire thing by json config anyway.

    parser = argparse.ArgumentParser(
        description='PyTorch partition as part of Async Pipeline')

    parser.add_argument('--rank', default=None,
                        type=int, help="Rank of worker")
    parser.add_argument('--local_rank', default=0,
                        type=int, help="Local rank of worker")

    parser.add_argument('--distributed_backend',
                        choices=['gloo', 'nccl', 'mpi'], default='mpi', type=str,
                        help='distributed backend to use')

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

    parser.add_argument("--weight_stashing", action="store_true",
                        default=False, help="Do weight Stashing")
    parser.add_argument("--log_frequency", type=int, default=100,
                        help="Print extra statistics every given number of batches")
    parser.add_argument("--max_buffers", type=int, default=2,
                        help="Maximal Number of async recv buffers. "
                        "With 1: it actaully means the recv is sync.(default=2 for best performance).")

    # TODO: option for weigth stashing just statistics.

    args = parser.parse_args()

    return args


def parse_json_config(args):
    assert(os.path.exists(args.config))

    with open(args.config, 'r') as f:
        output = json.load(f)

    # replace
    for key, value in output.items():
        if output.get(f'{key}_from_cmd', False):
            continue
        # if key == 'seed' and output.get('seed_from_cmd', False):
        #     continue
        # if key == 'bs_train' and output.get('bs_train_from_cmd', False):
        #     continue
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
        # Note this is overriden later.
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])


def assert_args(args):
    assert (args.epochs >= 1 or args.steps >= 1)
    assert(not (args.stage is None))


def create_comm_handler(args, comm_init_args, device) -> CommunicationHandlerBase:

    # get the parameters to create the comm handler
    handler_cls = get_auto_comm_handler_cls(args.distributed_backend, args.cpu)
    comm_handler = handler_cls(
        args.rank,
        args.local_rank,
        args.distributed_backend,
        args.world_size,
        args.num_stages,
        args.stage,
        *comm_init_args,
        args.cpu,
        args.num_chunks,
        device,
        GRAD_UGLY_SHAMEFUL_NAME="_grad",
        verbose=args.verbose_comm if hasattr(args, "verbose_comm") else False)

    return comm_handler


# target_tensor_names = {"target"}
def tensor_tags_from_config(config, target_tensor_names, num_chunks, GRAD_UGLY_SHAMEFUL_NAME="_grad"):

    def config_to_tuples_array(config):

        def config_to_tuples_generator(config):
            """ allows iterating with the tuple: (stage_id, inputs, outputs) """
            for i, v in config.items():
                yield i, v['inputs'], v['outputs']

        return np.array(list(config_to_tuples_generator(config)))

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


def get_my_send_recv_ranks(config, stage, stage_to_rank_map=None):

    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    # TODO: We assume this is same order with Alon's code/config, after poped some stuff.
    # Alon config is outside of the project, this is dangerous programing...
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

    return send_ranks, receive_ranks


def infer_dtypes_and_shapes(config, bs_train, bs_test, random_input_sample,
                            training_tensor_dtypes, training_tensor_shapes, eval_tensor_shapes,
                            just_for_stage=None):
    """
    Runs a sequential forward pass to determine:
        # training_tensor_dtypes
        # training_tensor_shapes
        # eval_tensor_shapes
        # TODO: eval_tensor_dtypes

    
    # FIXME: we don't want this pass to record statistic for batch norm!
    # TODO: maybe write this to some file and load from it if exists,
    #  to aviod doing this pass every time
    """
    assert(len(training_tensor_shapes) == len(training_tensor_dtypes))
    if not (just_for_stage is None):
        raise NotImplementedError()

    bs_train = to_tuple(bs_train)
    bs_test = to_tuple(bs_test)
    len_bs = len(bs_train)

    for i, v in config.items():
        partition = v['model']
        if i == 0:
            with torch.no_grad():
                a = partition(random_input_sample)
        else:
            with torch.no_grad():
                a = partition(*a)
        
        if (just_for_stage is None) or just_for_stage == i:
            # TODO: we need to actually go for i+1...
            outputs = v['outputs']
            dtypes = tuple(j.data.dtype for j in a)

            # Concatenate shapes with expected bs_train/bs_test
            # the batch size can be a collection (e.g (batch, seq_len) in NLP)
            # TODO: this assume that batch is first
            train_shapes = tuple(
                tuple(list(bs_train) + list(j.data.size()[len_bs:])) for j in a)
            eval_shapes = tuple(
                tuple(list(bs_test) + list(j.data.size()[len_bs:])) for j in a)

            training_tensor_dtypes.update(zip(outputs, dtypes))
            training_tensor_shapes.update(zip(outputs, train_shapes))
            eval_tensor_shapes.update(zip(outputs, eval_shapes))

        if just_for_stage == i:
            break

    return training_tensor_dtypes, training_tensor_shapes, eval_tensor_shapes


def create_distributed_communcation_context(args, config, stage,
                                            target_tensor_names=None,
                                            # training_tensor_dtypes={},
                                            # training_tensor_shapes={}, eval_tensor_shapes={},
                                            # random_input_sample=None,
                                            # bs_train=(1,),
                                            # bs_test=(1,),
                                            stage_to_rank_map=None):
    """
    Returns:
        target_tensor_names (input)
        tensor_tags
        receive_ranks
        send_ranks
        training_tensor_dtypes
        training_tensor_shapes
        eval_tensor_shapes
        ranks_in_previous_stage
        ranks_in_next_stage

    TODO:
        eval_tensor_dtypes
        support weight sharing
    """

    if target_tensor_names is None:
        target_tensor_names = set()
    
    tensor_tags, TOTAL_TAGS = tensor_tags_from_config(
        config, target_tensor_names, args.num_chunks, GRAD_UGLY_SHAMEFUL_NAME="_grad")

    send_ranks, receive_ranks = get_my_send_recv_ranks(
        config, stage, stage_to_rank_map=stage_to_rank_map)

    # Create:
    # ranks_in_previous_stage
    # ranks_in_next_stage

    # TODO: can create these by th econfig too.
    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    ranks_in_previous_stage = ranks_in_stage(
        stage - 1) if stage > 0 else []
    ranks_in_next_stage = ranks_in_stage(
        stage + 1) if stage < args.num_stages - 1 else []

    # Note that we don't need shapes for the comm, just the datatypes.
    comm_init_args = (receive_ranks,
                 send_ranks,
                 tensor_tags,
                 target_tensor_names,
                 ranks_in_previous_stage,
                 ranks_in_next_stage,
                 TOTAL_TAGS)

    return comm_init_args


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

    # TODO: this could be implemented by using the gap...
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
            optimizer_type, pred_mem, optimizer, scheduler=scheduler, nag_with_predictor=nag_with_predictor)
        return weight_predictor, nag_with_predictor
    else:
        raise NotImplementedError()


def hack_trainer_type_to_gap_aware(args):

    def hack():
        args.trainer['type'] += "_gap_aware"

    if hasattr(args, 'gap_aware'):
        # on = args.gap_aware['on']
        if args.gap_aware['policy'] == 'almost_last_partition':
            is_almost_last_partition = args.local_rank == args.world_size - 2

            # HACK: change trainer name
            if is_almost_last_partition:
                hack()
                return True

        elif args.gap_aware['policy'] == 'all_except_last':
            is_last_partition = args.local_rank == args.world_size - 1
            if not is_last_partition:
                hack()
                return True
        else:
            SUPPORTED_POLICIES = {'almost_last_partition', 'all_except_last'}
            raise ValueError(f"Uknown policy for GA {args.gap_aware['policy']}.\
                             suported policies are {SUPPORTED_POLICIES}")

    return False


def auto_file_name(args):
    assert hasattr(args, "auto_file_name")
    wp = args.weight_prediction['type'] if hasattr(
        args, "weight_prediction") else 'stale'
    ws = "ws_" if (hasattr(args, "weight_stashing")
                   and args.weight_stashing) else ""
    ga = "ga_" if hasattr(args, "gap_aware") else ""
    bs = f"bs_{args.bs_train * args.step_every}_"
    s = f'{args.model}_{args.dataset}_{wp}_{ws}{ga}{bs}seed_{args.seed}'
    args.out_filename = f"{args.out_filename}_{s}"
    print(f"Out File Name will be: {args.out_filename}")


def save_distributed_experiment(statistics, args, world_size, rank, local_rank, stage):

    def carefull_del(config, name):
        if name in config:
            del config[name]

    UN_NEEDED_ARGS = ['stage', 'rank', 'local_rank']

    if rank == world_size - 1:
        if statistics:
            fit_res = statistics.get_stats(stage)
            config = vars(args)

            # remove unneeded added args
            for name in UN_NEEDED_ARGS:
                carefull_del(config, name)

            save_experiment(args.out_filename, args.out_dir, config, fit_res)
    torch.distributed.barrier()

    # Update statistics one by one:
    for current_rank in reversed(range(world_size - 1)):
        if rank == current_rank:
            if statistics:
                my_fit_res = statistics.get_stats(stage)
                config, fit_res = load_experiment_for_update(
                    args.out_filename, args.out_dir)

                # Update just the fit res
                for k, v in my_fit_res.items():
                    if k not in fit_res:
                        fit_res[k] = v
                # save it
                save_experiment(args.out_filename,
                                args.out_dir, config, fit_res)

        torch.distributed.barrier()


def training_loop(args, logger, train_dl, test_dl, is_first_partition, partition, statistics, train_dl_len, test_dl_len):
    epochs = 0
    steps = 0
    logger.info(f"flush rate {args.flush_rate}")
    logger.info(f"Running for {args.epochs} epochs and {args.steps} steps")
    if (args.flush_rate >= 0):
        raise NotImplementedError()

    TRAIN_BATCHES_TO_RUN = getattr(args, "train_batches_limit", train_dl_len)
    TEST_BATCHES_TO_RUN = getattr(args, "test_batches_limit", test_dl_len)

    TRAIN_BATCHES_TO_RUN = train_dl_len if TRAIN_BATCHES_TO_RUN < 0 else TRAIN_BATCHES_TO_RUN
    TEST_BATCHES_TO_RUN = test_dl_len if TEST_BATCHES_TO_RUN < 0 else TEST_BATCHES_TO_RUN

    while epochs < args.epochs or args.epochs < 0:
        if args.steps > 0:
            TRAIN_BATCHES_TO_RUN = min(
                TRAIN_BATCHES_TO_RUN, args.steps - steps)

            # handle step every.
            reminder_to_drop = TRAIN_BATCHES_TO_RUN % args.step_every
            if reminder_to_drop:
                logger.info(
                    f"Drop {reminder_to_drop} steps for each epoch>={epochs}")
                TRAIN_BATCHES_TO_RUN -= reminder_to_drop
                if TRAIN_BATCHES_TO_RUN <= 0:
                    break

        did_train = False
        did_eval = False
        epoch_start_time = time.time()
        for TRAIN in [True, False]:
            logger.info(f"Running {'train' if TRAIN else 'eval'}")

            if TRAIN:
                if TRAIN_BATCHES_TO_RUN == 0:
                    continue
                # Set Dataloader
                # sets only to first (+last) partition
                if train_dl:
                    partition.set_dataloader(train_dl)
                # Start training
                partition.train()
                if statistics:
                    statistics.train()
                # if args.flush_rate > 0:
                #     for _ in range(0, TRAIN_BATCHES_TO_RUN, args.flush_rate):
                #         partition.run_until_flush(
                #             min(args.flush_rate, len(train_dl)))

                #     reminder = len(train_dl) % args.flush_rate
                #     if reminder > 0:
                #         partition.run_until_flush(reminder)
                # else:
                partition.run_until_flush(TRAIN_BATCHES_TO_RUN)

                did_train = True
                if args.local_rank == args.world_size - 1:
                    statistics.on_epoch_end()
                else:
                    statistics.non_latst_partition_on_epoch_end()
            else:  # EVAL
                # Set Dataloader
                # sets only to first (+last) partition
                if TEST_BATCHES_TO_RUN == 0:
                    continue
                if test_dl:
                    partition.set_dataloader(test_dl)
                partition.eval()

                if statistics:
                    statistics.eval()

                with torch.no_grad():  # TODO maybe remove this?
                    partition.run_forward_until_flush(TEST_BATCHES_TO_RUN)

                did_eval = True
                if args.local_rank == args.world_size - 1:
                    statistics.on_epoch_end()

        epochs += 1
        if did_train:
            steps += math.ceil(TRAIN_BATCHES_TO_RUN / args.step_every)

        # if is_last_partition
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
            logger.info(
                f"Finished all steps. Total steps:{steps}, rank:{args.local_rank}")
            break  # steps condition met


def get_dataloaders(args, explicit_seperated_dataset=False):
    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    else:
        dl_kw['pin_memory'] = True

    dl_kw['num_workers'] = args.num_data_workers
    dl_kw['drop_last'] = True

    # if args.task == 'cv_sep'
    if explicit_seperated_dataset:
        train_dl, test_dl = get_seperate_just_x_or_y_train_test_dl_from_args(args, verbose=False, **dl_kw)
    else:
        train_dl, test_dl = simplified_get_train_test_dl_from_args(
            args, verbose=False, **dl_kw)

    return train_dl, test_dl


def get_device(args):
    if hasattr(args, "stage_to_device_map"):
        stage_to_device_map = args.stage_to_device_map
        cuda_device_id = stage_to_device_map[args.stage]
        device = torch.device('cpu' if args.cpu else f"cuda:{cuda_device_id}")
    else:
        device = torch.device('cpu' if args.cpu else f"cuda:{args.local_rank}")
    return device


def main():
    args = parse_cli()
    parse_json_config(args)
    parse_env_vars(args)
    args.world_size = get_world_size(args.distributed_backend)

    # TODO: idealy we want to choose device here, but we moved it down.

    # Set Random Seed
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31)
    # FIXME: I susspect there is a problem here because it does it on it on ALL GPUs.
    # should probably hide with CUDA VISABLE DEVICES, 
    # or do it just for a single GPU:
    # torch._C.default_generator.manual_seed(int(args.seed))
    # torch.cuda.manual_seed(int(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if hasattr(args, "cudnn_benchmark") and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Get partitioning config
    configs = models.get_partitioning(args.model, model_instance=None)
    configs.pop('model inputs')  # We don't use thous.
    configs.pop('model outputs')  # We don't use thous.

    NO_DP = True
    # TODO: make it nicer.
    args.stage = None
    if NO_DP:
        args.num_stages = len(configs)
        args.stage = args.local_rank
        is_first_partition = args.local_rank == 0
        is_last_partition = args.local_rank == len(configs) - 1
        # args.num_ranks = len(configs)
    else:
        raise NotImplementedError()

    # torch.device('cpu' if args.cpu else f"cuda:{args.local_rank}")
    device = get_device(args)
    if not args.cpu:
        torch.cuda.set_device(device)

    args.step_every = getattr(args, "step_every", 1)
    args.base_lr_batch_size = getattr(
        args, "base_lr_batch_size", args.bs_train)

    assert_args(args)

    if args.debug:
        # TODO: by specific ranks
        import ptvsd
        port = 3000 + args.local_rank
        address = ('127.0.0.1', port)
        print(f"-I- waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()

    logger = FileLogger(args.logdir, global_rank=args.rank,
                        local_rank=args.local_rank, name='msnag', world_size=args.world_size,
                        name_prefix=args.out_filename)  # TODO: real name

    partition_using_gap_aware = hack_trainer_type_to_gap_aware(args)
    if partition_using_gap_aware:
        logger.info(f"Stage {args.stage} will use Gap Aware")

    # Get dataloaders
    train_dl, test_dl = get_dataloaders(args)

    ######################################## Start OF UGLY BLOCK ########################################
    # TODO: do the following block generically and automatically using tasks, or alon's code.
    x, y = next(iter(train_dl))
    bs_train = to_tuple(args.bs_train)
    bs_test = to_tuple(args.bs_test)

    BASE_INPUT_SHAPE = x.shape[1:]
    BASE_TARGET_SHAPE = y.shape[1:]

    # TODO formalize with function according to dataset/task
    USE_TARGET = not ('_sep' in args.task)
    target_tensor_names = {}
    training_tensor_dtypes = {"input0": x.dtype}
    training_tensor_shapes = {"input0": (*bs_train, *BASE_INPUT_SHAPE)}
    eval_tensor_shapes = {"input0": (*bs_test, *BASE_INPUT_SHAPE)}

    if USE_TARGET:
        target_tensor_names = {"target"}
        training_tensor_dtypes["target"] = y.dtype
        training_tensor_shapes["target"] = (*bs_train, *BASE_TARGET_SHAPE)
        eval_tensor_shapes["target"] = (*bs_test, *BASE_TARGET_SHAPE)

    SAMPLE_BATCH_SIZE = 1  # Smallest batch as possible.
    random_input_sample = torch.randn(SAMPLE_BATCH_SIZE, *BASE_INPUT_SHAPE)

    del x
    del y

    # eval_tensor_shapes, training_tensor_shapes, target_tensor_names, random_input_sample

    comm_init_args = \
        create_distributed_communcation_context(args, configs, args.stage,
                                                target_tensor_names=target_tensor_names,
                                                stage_to_rank_map=None)

    comm_handler = create_comm_handler(args, comm_init_args, device)

    (training_tensor_dtypes,
     training_tensor_shapes,
     eval_tensor_shapes) = \
        infer_dtypes_and_shapes(
        configs, bs_train, bs_test, random_input_sample,
        training_tensor_dtypes, training_tensor_shapes, eval_tensor_shapes, just_for_stage=None)

    ######################################## END OF UGLY BLOCK ########################################

    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer['type'])
    task_cls = AVAILABLE_TASKS.get(args.task)
    optimizer_cls = AVAILBALE_OPTIMIZERS.get(args.optimizer_type)
    statistics = AVAILBALE_STATS.get(args.statistics)
    assert not (statistics is None)
    work_scheduler = AVAILABLE_WORK_SCHEDULERS.get(args.work_scheduler)

    # Init the partition manager
    partition = SinglePartitionManager(
        args.stage,
        configs, configs[args.stage]['model'],
        comm_handler, work_scheduler,
        training_tensor_shapes, eval_tensor_shapes, training_tensor_dtypes,
        device, is_last_partition, is_first_partition,
        log_frequency=args.log_frequency,
        max_buffers=args.max_buffers,
        step_every=args.step_every
    )

    if hasattr(args, "ddp_sim_num_gpus") and args.ddp_sim_num_gpus > 1:
        print(
            f"-I- simulating DDP accuracy with {args.ddp_sim_num_gpus} (DDP) GPUs per stage")
        dp_sim.convert_to_num_gpus(partition.partition, args.ddp_sim_num_gpus)

    # After the partition is on its device:
    # Set optimizer
    optimizer = optimizer_cls(
        partition.partition.parameters(), **args.optimizer['args'])

    # Create micro-batching (big batch) manger
    # big_batch_mgr = BigBatchManager(
    #     args.step_every, optimizer, args.base_lr_batch_size, args.bs_train) if args.step_every > 1 else None
    # if len(train_dl) % args.step_every != 0:
    #     reminder_to_drop = len(train_dl) % args.step_every
    #     raise NotImplementedError(
    #         f"len(train_dl):{len(train_dl)}, args.step_every:{args.step_every}")
    if args.flush_rate > 0 and args.flush_rate < args.step_every:
        raise NotImplementedError()

    # partition.set_big_batch_mgr(big_batch_mgr)
    # Set Scheduler
    # TODO: scheduler for sched aware prediction
    scheduler = get_scheduler(args, optimizer)

    # Set Trainer (and Gap Aware)
    trainer_extra_args = args.trainer['args']
    if hasattr(trainer_cls, "HAS_GAP_AWARE"):
        gap_aware = get_gap_aware(args, optimizer)
        trainer = trainer_cls(gap_aware, partition.partition, optimizer=optimizer,
                              scheduler=scheduler, statistics=statistics, **trainer_extra_args)
    else:
        gap_aware = None
        trainer = trainer_cls(partition.partition, optimizer=optimizer,
                              scheduler=scheduler, statistics=statistics, **trainer_extra_args)

    partition.set_trainer(trainer)
    partition.set_lr_scheduler(scheduler)

    # Set Weight predictor
    weight_predictor, nag_with_predictor = get_weight_predictor(
        args, optimizer, scheduler=scheduler)
    if weight_predictor:
        partition.set_weight_predictor(weight_predictor, nag_with_predictor)

    # Set Weight Stashing
    weight_stasher = WeightStasher(optimizer, args.step_every) if hasattr(
        args, "weight_stashing") and args.weight_stashing else None
    if weight_stasher and not is_last_partition:
        partition.set_weight_stasher(weight_stasher)

    # Set Task
    task = task_cls(device, is_last_partition, is_first_partition)
    partition.set_task(task)

    if hasattr(args, "auto_file_name"):
        auto_file_name(args)

    train_dl_len, test_dl_len = len(train_dl), len(test_dl)
    # Try getting seperate X,Y dataloaders
    if is_first_partition or is_last_partition:
        if "_sep" in args.task:
            train_dl, test_dl = get_dataloaders(args, explicit_seperated_dataset=True)
    else:
        train_dl, test_dl = None, None

    # Main Training Loop

    training_loop(args, logger, train_dl, test_dl,
                  is_first_partition, partition, statistics, train_dl_len, test_dl_len)

    # Synchronize and save statistics from all partitions
    save_distributed_experiment(
        statistics, args, args.world_size, args.rank, args.local_rank, args.stage)


if __name__ == "__main__":

    print(f"Using {torch.get_num_threads()} Threads")
    main()
