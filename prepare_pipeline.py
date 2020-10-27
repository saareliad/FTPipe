import gc
import warnings
from enum import Enum, auto
from typing import Type

import torch
from torch.utils.data import DataLoader

import models
import optimizers.lr_scheduler
# from data import get_dataloaders  # Commented out because can't install t5 on windows.
from experiments.experiments import ArgsStasher, auto_file_name
from misc.filelogger import FileLogger
from models import parse_config
from optimizers import AVAILBALE_OPTIMIZERS
from pipeline import CommunicationHandlerBase, get_auto_comm_handler_cls
from pipeline import TrueWeightsStorage
from pipeline import dp_sim
from pipeline.communication.multiprocessing import MultiprocessingCommunicationHandler
from pipeline.data_propagation import get_propagator_cls
from pipeline.gap_aware import (get_sgd_gap_aware_cls, get_adam_gap_aware_cls,
                                get_adamw_gap_aware_cls)
from pipeline.partition_manager import (SinglePartitionManager, GPipePartitionManager)
from pipeline.statistics import get_statistics  # , Stats
from pipeline.training import AVAILABLE_TRAINERS
from pipeline.weight_prediction import get_sched_predictor
from pipeline.weight_prediction import get_weight_predictor as get_weight_predictor_partial
from pipeline.weight_stashing import WeightStasher
# TODO: migrate to `register_xxx()` convention
from pipeline.work_schedulers import get_work_scheduler


# from data import AVAILABLE_DATASETS

def get_trainer_cls(args):
    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer['type'])
    return trainer_cls


def is_huggingface_transformer(args):
    if getattr(args, "is_huggingface_transformer", False):
        return True
    return args.model in models.transformers_cfg.MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS.keys()


def create_comm_handler(args, comm_init_args,
                        device) -> CommunicationHandlerBase:
    # get the parameters to create the comm handler
    handler_cls = get_auto_comm_handler_cls(args.distributed_backend, args.cpu)
    comm_handler = handler_cls(args.rank,
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
                               verbose=getattr(args, "verbose_comm", False))

    return comm_handler


def create_comm_handler_v2(args, comm_init_args, device,
                           v2_args) -> CommunicationHandlerBase:
    handler_cls = MultiprocessingCommunicationHandler
    comm_handler = handler_cls(
        *v2_args,
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


def get_lr_scheduler(args, optimizer):
    # NOTE: in some optimizers version we can benefit from lr=0 (build momentum in place)
    if hasattr(args, "lr_scheduler"):
        attr = getattr(args, 'lr_scheduler')
        preproc_lr_scheduler_args(args)
        scheduler_cls = get_lr_scheduler_class(args)
        scheduler = scheduler_cls(optimizer, **attr['args'])
        return scheduler


def preproc_lr_scheduler_args(args):
    attr = getattr(args, 'lr_scheduler')
    preproc_args = attr.get("preproc_args", None)
    if preproc_args:
        # auto-calculate numbers like num_training_steps, num_warmup_steps
        for arg_name, preproc_command in preproc_args.items():
            if preproc_command == "epochs_to_steps":
                if args.steps > 0:
                    raise NotImplementedError(
                        "Expected to be limited by number of epochs")

                # Get the given number of epochs
                given_epochs = attr['args'][arg_name]
                if given_epochs < 0:
                    # Taking from epoch args
                    if args.epochs < 0:
                        raise ValueError(
                            "Expected a concrete number of epochs")
                    given_epochs = args.epochs

                # Translate epochs to steps
                num_steps = args.steps_per_epoch * given_epochs
                attr['args'][arg_name] = num_steps
                print(
                    f"preprocessed {arg_name} from {given_epochs} epochs to {num_steps} steps."
                )
            elif preproc_command == "ratio_from_num_training_steps":
                num_steps = attr['args']['num_training_steps']
                given_ratio = attr['args'][arg_name]
                assert given_ratio >= 0 and given_ratio <= 1
                warmup_steps = int(given_ratio * num_steps)
                attr['args'][arg_name] = warmup_steps
                print(
                    f"preprocessed {arg_name} from ratio {given_ratio} to {warmup_steps} steps."
                )
            else:
                raise NotImplementedError(
                    f"Unsupported preprocess argument {preproc_command}")


def get_lr_scheduler_class(args):
    attr = getattr(args, 'lr_scheduler')
    if attr['type'] in optimizers.lr_scheduler.AVAILABLE_LR_SCHEDULERS:
        scheduler_cls = getattr(optimizers.lr_scheduler, attr['type'])
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, attr['type'])
    return scheduler_cls


def get_sched_aware_stuff(args):
    attr = getattr(args, 'lr_scheduler')
    scheduler_cls = get_lr_scheduler_class(args)
    sched_aware_stuff = (scheduler_cls, attr['args'])
    return sched_aware_stuff


def get_gap_aware(args, optimizer):
    if not hasattr(args, 'gap_aware'):
        return None
    gap_aware_args = getattr(args, 'gap_aware')['args']
    optimizer_type = getattr(args, 'optimizer')['type']

    # TODO: this could be implemented by using the gap...
    if not optimizer_type == 'sgd1' and not getattr(args, 'weight_stashing',
                                                    False):  # pytorch
        raise NotImplementedError()

    if 'sgd' in optimizer_type:
        gap_aware_cls = get_sgd_gap_aware_cls(optimizer_type)
    elif 'adam' == optimizer_type:
        gap_aware_cls = get_adam_gap_aware_cls()
    elif 'adamw' == optimizer_type:
        gap_aware_cls = get_adamw_gap_aware_cls()
    elif 'adafactor' == optimizer_type:
        raise NotImplementedError("WIP")
    else:
        raise NotImplementedError

    return gap_aware_cls(optimizer, **gap_aware_args)


def try_replace_prediction_with_nesterov(args):
    # If last partition: just use nesterov.
    optimizer_type = getattr(args, 'optimizer')['type']
    if "sgd" in optimizer_type and getattr(
            args, "nesterov_set_for_last_partition", False):
        tmp = args.optimizer['args']
        if not tmp.get('nesterov', False):
            pred = getattr(args, 'weight_prediction', None)

            if (pred is not None):
                tmp['nesterov'] = True
                pred['args']['nag_with_predictor'] = False
                args.nesterov_set_for_last_partition = True
                print("-I- Setting nesterov=True for last partition")
                res = getattr(args, 'weight_prediction')
                # For naming purposes
                ArgsStasher.stash_to_args(args, replaced_key='weight_prediction', old_value=res)
                delattr(args, 'weight_prediction')


def get_weight_predictor(args,
                         optimizer,
                         scheduler=None,
                         true_weights_storage=None):
    """
        Returns:
            weight_predictor,
            nag_with_predictor: bool
    """
    assert (true_weights_storage is not None
            )  # TODO: should be normal argument when its stable
    if not hasattr(args, 'weight_prediction'):
        return None, None

    optimizer_type = getattr(args, 'optimizer')['type']
    pred = getattr(args, 'weight_prediction')
    pred_mem = pred['args']['pred_mem']
    pred_type = pred['type']
    nag_with_predictor = pred['args'].get('nag_with_predictor', False)

    assert (pred_mem in {"clone", "calc"})
    assert (pred_type in {"msnag", "aggmsnag"})
    assert ('sgd' in optimizer_type or 'adam' in optimizer_type)

    sched_predictor = get_sched_aware_predictor(args, optimizer, scheduler)

    weight_predictor = get_weight_predictor_partial(
        optimizer_type,
        pred_mem,
        pred_type,
        optimizer,
        scheduler=sched_predictor,
        nag_with_predictor=nag_with_predictor,
        true_weights_storage=true_weights_storage,
        sched_predictor=sched_predictor)

    assert weight_predictor is not None
    return weight_predictor, nag_with_predictor


def get_sched_aware_predictor(args, optimizer, scheduler):
    optimizer_type = getattr(args, 'optimizer')['type']
    pred = getattr(args, 'weight_prediction')
    sched_predictor = None
    if pred['args'].get("sched_aware", False):
        print("-I- using sched aware weight prediction")
        assert scheduler is not None
        sched_aware_stuff = get_sched_aware_stuff(args)
        assert sched_aware_stuff is not None
        scheduler_class = sched_aware_stuff[0]
        scheduler_kw = sched_aware_stuff[1]
        sched_predictor = get_sched_predictor(optimizer, scheduler_class,
                                              **scheduler_kw)
        sched_predictor.patch_scheduler(scheduler)
        assert 'adam' in optimizer_type  # Remove after we implement for sgd.
    return sched_predictor


def get_ngpus_per_node(args):
    nnodes = args.nnodes
    if not hasattr(args, "ngpus_per_node"):
        # same number across all nodes
        if args.world_size % nnodes != 0:
            raise NotImplementedError()
        ngpus_per_node = [args.world_size // nnodes] * nnodes
    else:
        ngpus_per_node = args.ngpus_per_node
    assert len(ngpus_per_node) == nnodes
    return ngpus_per_node


def get_device_for_rank(args, rank, local_rank):
    nnodes = args.nnodes
    ngpus_per_node = get_ngpus_per_node(args)

    # Infer local device ID.
    if hasattr(args, "stage_to_device_map"):
        stage_to_device_map = args.stage_to_device_map
        cuda_device_id = stage_to_device_map[rank]

        # global to local device id
        if nnodes > 1:
            for node_idx, x in enumerate(ngpus_per_node):
                if cuda_device_id >= x:
                    cuda_device_id -= x
                else:
                    break
            else:
                raise ValueError(
                    f"Can't determine device index. rank={rank}, stage_to_device_map={stage_to_device_map}, global_device_id={cuda_device_id},  nnodes={nnodes}, ngpus_per_node={ngpus_per_node}")

        local_device_id = cuda_device_id
    else:
        local_device_id = local_rank

    # Get device
    device = torch.device('cpu' if args.cpu else f"cuda:{local_device_id}")
    return device


def get_rank_to_device_map(args):
    if args.nnodes == 1:
        local_ranks = list(range(args.world_size))
    else:

        ngpus_per_node = get_ngpus_per_node(args)
        local_ranks = list()
        for n in ngpus_per_node:
            local_ranks.extend(range(n))

    return {
        rank: get_device_for_rank(args, rank, local_rank)
        for rank, local_rank in zip(range(args.world_size), local_ranks)
    }


def hack_trainer_type_to_gap_aware(args, stage_depth=None):
    """ replaces TRAINER with TRAINER_gap_aware,
        according to parsed policy
        SUPPORTED_POLICIES = {
            'almost_last_partition', 
            'all_except_last',
            'all_except_last_two'
            }
        # TODO: policy for max delay 1
        # TODO: policy with staleness limit
    """

    def hack():
        args.trainer['type'] += "_gap_aware"

    if hasattr(args, 'gap_aware'):

        if stage_depth is None:
            is_zero_staleness_stage = args.local_rank == args.world_size - 1
            is_one_staleness_stage = args.local_rank == args.world_size - 2

        else:
            is_zero_staleness_stage = stage_depth == 0
            is_one_staleness_stage = stage_depth == 1
            warnings.warn("Assuming no grad accumulation and no staleness limit...")

        if args.gap_aware['policy'] == 'almost_last_partition':
            # HACK: change trainer name
            if is_one_staleness_stage:
                hack()
                return True
        elif args.gap_aware['policy'] == 'all_except_last':
            if not is_zero_staleness_stage:
                hack()
                return True
        elif args.gap_aware['policy'] == 'all_except_last_two':
            if (not is_zero_staleness_stage) and (not is_one_staleness_stage):
                hack()
                return True
        else:
            SUPPORTED_POLICIES = {
                'almost_last_partition', 'all_except_last',
                'all_except_last_two'
            }
            raise ValueError(
                f"Unknown policy for GA {args.gap_aware['policy']}.\
                             supported policies are {SUPPORTED_POLICIES}")
    return False


def get_optimizer_cls(args, has_gap_aware):
    optimizer_type = args.optimizer['type']

    # Optimizers which also record square step size.
    if has_gap_aware and optimizer_type in {'adam', 'adamw'}:
        optimizer_type += '_record_step'

    optimizer_cls = AVAILBALE_OPTIMIZERS.get(optimizer_type)
    assert optimizer_cls is not None, f"{optimizer_type} not in {AVAILBALE_OPTIMIZERS.keys()}"
    return optimizer_cls


def tuplify(listything):
    if isinstance(listything, list):
        return tuple(map(tuplify, listything))
    if isinstance(listything, dict):
        return {k: tuplify(v) for k, v in listything.items()}
    return listything


def get_optimizer(args, optimizer_cls, parameters):
    assert isinstance(parameters, list)
    if len(parameters) == 0:
        if not getattr(args, "allow_stateless", False):
            raise ValueError(f"Got stateless partition {args.stage}")

    # HACK: tuplify all optimizer paramerets, just in case. [0.9, 0.98] -> (0.9, 0.98)
    # https://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json
    tuplified_opt_args = tuplify(args.optimizer['args'])
    optimizer = optimizer_cls(parameters, **tuplified_opt_args)
    return optimizer


def preproc_data(args, cache=None, save_cache=True):
    # TODO: currently runs as an outside for loop for all stages

    # Parse partitioning config and requires args
    print(f"Loading partitioned model and dataset...")

    if cache is None:
        handler = models.AVAILABLE_MODELS.get(args.model)
        if save_cache:
            cache = handler
    else:
        handler = cache

    parsed_config = parse_config.PartitioningConfigParser(
        args.model,
        args.rank,
        args.bs_train,
        args.bs_test,  # NOTE: changed name
        handler=handler,
        send_target_in_pipe=("_nonsep" in args.data_propagator),
        prefer_seq_sends=getattr(args, "prefer_seq_sends", True))

    dataset_keywords = {}
    parsed_config.load_model(handler=handler, bs_train=args.bs_train, rank=args.rank)
    extra_kw = handler.get_extra()
    if isinstance(extra_kw, dict):
        dataset_keywords.update(extra_kw)
    # NOTE: it can be saved in cache
    # delete to save mem, in contains original model
    del handler

    pipe_config = parsed_config.pipe_config
    args.num_stages = parsed_config.num_stages
    args.stage = parsed_config.stage_id
    from data import get_dataloaders
    get_dataloaders(
        args,
        pipe_config=pipe_config,
        dataset_keywords=dataset_keywords)

    return cache


def prepare_pipeline(args, shared_ctx=None, COMM_VERSION=1):
    is_gpipe = "gpipe" == args.work_scheduler.lower()

    if args.is_multiprocessing_worker:
        # multiprocessing communication handler
        COMM_VERSION = 2

    # get work scheduler

    # set device
    local_rank_to_device_map = get_rank_to_device_map(args)
    device = local_rank_to_device_map[args.local_rank]
    if not args.cpu:
        torch.cuda.set_device(device)

    # Parse partitioning config and requires args
    # TODO: some easier way to get original model and the config used during partitioning (WIP)
    print(f"Loading partitioned model and dataset...")
    handler = models.AVAILABLE_MODELS.get(args.model)

    parsed_config = parse_config.PartitioningConfigParser(
        args.model,
        args.rank,
        args.bs_train,
        args.bs_test,  # NOTE: changed name
        handler=handler,
        send_target_in_pipe=("_nonsep" in args.data_propagator),
        prefer_seq_sends=getattr(args, "prefer_seq_sends", True))
    pipe_config = parsed_config.pipe_config

    # ini distributed
    args.num_stages = parsed_config.num_stages
    args.stage = parsed_config.stage_id
    comm_init_args = parsed_config.comm_init_args()

    assert (args.epochs >= 1 or args.steps >= 1)
    assert (not (args.stage is None))
    # FIXME: real name
    logger = FileLogger(args.logdir,
                        global_rank=args.rank,
                        local_rank=args.local_rank,
                        name='msnag',
                        world_size=args.world_size,
                        name_prefix=args.out_filename)

    # Comm handler
    if COMM_VERSION == 1:
        comm_handler = create_comm_handler(args, comm_init_args, device)
        comm_handler.init_process_group()
    elif COMM_VERSION == 2:
        # Multiprocessing
        # TODO stage_to_device_map is currently unused, local_rank_to_device_map is used instead.
        # stage_to_device_map will be used when combining MPI overlay
        stage_to_device_map = []  # TODO ^
        v2_args = (shared_ctx, stage_to_device_map, local_rank_to_device_map)
        comm_handler = create_comm_handler_v2(args, comm_init_args, device,
                                              v2_args)
    else:
        raise NotImplementedError("In progress")

    work_scheduler = get_work_scheduler(args, pipe_config=pipe_config)

    dataset_keywords = {}
    # Do heavy ram part one by one to save memory.
    # TODO this can be done better, e.g by local ranks or in parallel.
    for i in range(args.world_size):
        if getattr(args, "load_model_one_by_one", False):
            print(f"loading the model rank by rank to save host RAM {i + 1}/{args.world_size}")
            torch.distributed.barrier()
        if i == args.rank:
            parsed_config.load_model(handler=handler, bs_train=args.bs_train, rank=args.rank)
            extra_kw = handler.get_extra()
            if isinstance(extra_kw, dict):
                dataset_keywords.update(extra_kw)
            # delete to save mem, in contains original model
            del handler
            gc.collect()

    training_tensor_dtypes = parsed_config.training_tensor_dtypes
    eval_tensor_shapes = parsed_config.eval_tensor_shapes
    training_tensor_shapes = parsed_config.training_tensor_shapes

    # NOTE: here its the sliced model.
    model = parsed_config.model

    model.device = device

    stage_depth = pipe_config.get_depth_for_stage(args.stage)
    pipeline_depth = pipe_config.pipeline_depth
    args.pipeline_depth = pipeline_depth

    # we assume last stage is the output
    is_last_partition = args.stage == args.num_stages - 1  # or stage_depth == 0
    is_first_partition = args.stage == 0  # or stage_depth == pipeline_depth - 1

    is_zero_staleness_stage = stage_depth == 0 if not is_gpipe else True

    eval_tensor_dtypes = training_tensor_dtypes  # HACK, TODO
    # Get dataloaders needed for this stage
    from data import get_dataloaders
    train_dl, eval_dl, samplers, extra = get_dataloaders(
        args,
        pipe_config=pipe_config,
        dataset_keywords=dataset_keywords)

    del dataset_keywords

    # instead of loading dl on every device just to get its length
    # we synchronize length as a message, from first stage
    (last_batch_diff_eval,
     last_batch_diff_train,
     eval_dl_len,
     train_dl_len) = synchronize_dataloaders_length(args,
                                                    is_first_partition,
                                                    logger,
                                                    eval_dl,
                                                    train_dl)
    if last_batch_diff_train > 0:
        last_batch_train_shapes = parsed_config.get_shapes(
            last_batch_diff_train)
    else:
        last_batch_train_shapes = None

    if last_batch_diff_eval > 0:
        last_batch_eval_shapes = parsed_config.get_shapes(last_batch_diff_eval)
    else:
        last_batch_eval_shapes = None

    # Get expected training steps:
    if args.epochs > 0 and args.steps < 0:
        steps_per_epoch = train_dl_len // args.step_every
        # TODO: and policy is proportional
        if train_dl_len % args.step_every > 0:
            STEP_EVERY_SMALLER_LAST_BATCH_POLICY = getattr(args, "STEP_EVERY_SMALLER_LAST_BATCH_POLICY",
                                                           SmallerLastBatchPolicy.ProportionalStep)
            if STEP_EVERY_SMALLER_LAST_BATCH_POLICY == SmallerLastBatchPolicy.ProportionalStep:
                steps_per_epoch += 1
        args.steps_per_epoch = steps_per_epoch  # used later if preproc lr scheduler from epochs
        expected_training_steps = steps_per_epoch * args.epochs
    elif args.epochs < 0 and args.steps > 0:
        expected_training_steps = args.steps
    else:
        raise NotImplementedError("Missing steps or epochs limit")

    # TODO: this is unused
    args.expected_training_steps = expected_training_steps

    buffers_ctx = (
        training_tensor_shapes,
        eval_tensor_shapes,
        training_tensor_dtypes,
        eval_tensor_dtypes,
        last_batch_train_shapes,
        last_batch_eval_shapes,
        args.max_buffers,
        args.keep_buffers_alive,
    )

    comm_handler.init_buffers_ctx(buffers_ctx)

    ##############################
    # Until here its common,
    # To GPipe too.
    ##############################

    # Will automatically change trainer to gap aware compatible trainer
    partition_using_gap_aware = False
    if not is_gpipe:
        partition_using_gap_aware = hack_trainer_type_to_gap_aware(args, stage_depth)

    if partition_using_gap_aware:
        logger.info(f"Stage {args.stage} will use Gap Aware")

    trainer_cls = get_trainer_cls(args)
    propagator_cls = get_propagator_cls(args)
    optimizer_cls = get_optimizer_cls(args, partition_using_gap_aware)
    statistics = get_statistics(args.statistics,
                                is_last_partition=is_last_partition)
    assert not (statistics is None)

    # Gap aware penalty just for the loss
    gap_aware_just_loss = False
    if not is_gpipe:
        gap_aware_just_loss = getattr(args, 'gap_aware_just_loss', False)
        if gap_aware_just_loss:
            if is_zero_staleness_stage:
                gap_aware_just_loss = False
            else:
                if args.no_recomputation:
                    raise NotImplementedError(
                        "gap_aware_just_loss works only with recomputation on")

    # Init the partition manager itself, warping the model and loading it to device.
    # select a partition manager
    if is_gpipe:
        partition_mgr_cls = GPipePartitionManager
    else:
        partition_mgr_cls = SinglePartitionManager

    partition_mgr_cls: Type[SinglePartitionManager]
    partition = partition_mgr_cls(
        args.stage,
        stage_depth,
        pipeline_depth,
        args.num_stages,
        model,
        comm_handler,
        work_scheduler,
        device,
        is_last_partition,
        is_first_partition,
        log_frequency=args.log_frequency,
        step_every=args.step_every,
        use_recomputation=(not args.no_recomputation),
        gap_aware_just_loss=gap_aware_just_loss,
        weight_stashing_just_for_stats=getattr(
            args, "weight_stashing_just_for_stats", False),
        disable_clone_inputs=args.is_multiprocessing_worker,
        req_grad=parsed_config.req_grad,
        # outputs_req_grad=parsed_config.outputs_req_grad
    )

    # support for simulating stage replication (dev)
    if hasattr(args, "ddp_sim_num_gpus") and args.ddp_sim_num_gpus > 1:
        print(
            f"-I- simulating DDP accuracy with {args.ddp_sim_num_gpus} (DDP) GPUs per stage"
        )
        dp_sim.convert_to_num_gpus(partition.partition, args.ddp_sim_num_gpus)

    # After the partition is on its device:
    # Set optimizer
    optimizer_grouped_parameters = get_optimizer_parameter_groups(args, partition)

    # if we replace wp with nesterov, we save the wp arg, and set it back for config and auto experiment naming.
    if not is_gpipe and is_zero_staleness_stage:
        try_replace_prediction_with_nesterov(args)

    optimizer = get_optimizer(args, optimizer_cls,
                              optimizer_grouped_parameters)

    if 0 < args.flush_rate < args.step_every:
        raise NotImplementedError()

    # Get Learning Rate Scheduler
    scheduler = get_lr_scheduler(args, optimizer)

    # Set Trainer (and Gap Aware)
    trainer_kwds = dict(model=partition.partition,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        statistics=statistics,
                        step_every=args.step_every)
    trainer_kwds.update(args.trainer['args'])
    # NOTE: With hack_trainer_type_to_gap_aware  we modified trainer type if needed.
    if getattr(trainer_cls, "HAS_GAP_AWARE", False):
        gap_aware = get_gap_aware(args, optimizer)
        trainer = trainer_cls(gap_aware, **trainer_kwds)
        partition.set_gap_aware(gap_aware)
        assert not is_gpipe
    else:
        gap_aware = None
        trainer = trainer_cls(**trainer_kwds)

    if extra:
        extra(trainer)

    partition.set_trainer(trainer)
    partition.set_lr_scheduler(scheduler)

    if not is_gpipe:
        # True weights storage
        true_weights_storage = TrueWeightsStorage(optimizer)
        partition.set_true_weights_storage(true_weights_storage)

        # Set Weight predictor
        weight_predictor, nag_with_predictor = get_weight_predictor(
            args,
            optimizer,
            scheduler=scheduler,
            true_weights_storage=true_weights_storage,
        )
        if weight_predictor:
            partition.set_weight_predictor(weight_predictor,
                                           nag_with_predictor)
            logger.info(f"Stage {args.stage} will use Weight Predictor")

        # Set Weight Stashing
        if getattr(args, "weight_stashing", False):
            if not is_zero_staleness_stage:

                has_weight_predictor = weight_predictor is not None
                if has_weight_predictor:
                    using_clone_weight_predictor = args.weight_prediction[
                                                       'args']['pred_mem'] == 'clone'
                else:
                    using_clone_weight_predictor = False

                weight_stasher = WeightStasher(
                    optimizer,
                    step_every=args.step_every,
                    has_weight_predictor=has_weight_predictor,
                    true_weights_storage=true_weights_storage,
                    using_clone_weight_predictor=using_clone_weight_predictor)
                partition.set_weight_stasher(weight_stasher)

        if gap_aware_just_loss:
            assert (getattr(args, "weight_stashing", False))

    # Set Data propagator
    propagator = propagator_cls(device, is_last_partition, is_first_partition, args.stage, pipe_config)
    partition.set_data_propagator(propagator)

    if hasattr(args, "auto_file_name"):
        auto_file_name(args)

    return (logger, train_dl, eval_dl, is_first_partition, is_last_partition,
            partition, statistics, train_dl_len, eval_dl_len, samplers)


def synchronize_dataloaders_length(args, is_first_partition: bool, logger, eval_dl: DataLoader, train_dl: DataLoader):
    if args.rank == 0:
        assert is_first_partition
        train_dl_len, eval_dl_len = len(train_dl), len(eval_dl)
        train_dataset_len, eval_dataset_len = len(train_dl.dataset), len(
            eval_dl.dataset)
        # TODO: support replicated

        last_batch_diff_train = train_dataset_len % args.bs_train if not train_dl.drop_last else 0
        last_batch_diff_eval = eval_dataset_len % args.bs_test if not eval_dl.drop_last else 0
        d = dict(train_dataset_len=train_dataset_len, eval_dataset_len=eval_dataset_len,
                 train_dl_len=train_dl_len, eval_dl_len=eval_dl_len,
                 last_batch_diff_train=last_batch_diff_train, last_batch_diff_eval=last_batch_diff_eval)
        logger.info(f"Synchronized: {d}")
        data = [
            train_dl_len, eval_dl_len, last_batch_diff_train,
            last_batch_diff_eval
        ]
        data = torch.tensor(data, dtype=torch.long)
    else:
        data = torch.zeros(4, dtype=torch.long)
    torch.distributed.broadcast(data, 0)
    train_dl_len = data[0].item()
    eval_dl_len = data[1].item()
    last_batch_diff_train = data[2].item()
    last_batch_diff_eval = data[3].item()

    return last_batch_diff_eval, last_batch_diff_train, eval_dl_len, train_dl_len


def get_optimizer_parameter_groups(args, partition):
    # If is transformer, use grouped parameters.
    if is_huggingface_transformer(args):
        # No weight decay for some parameters.
        model = partition.partition
        opt_args = args.optimizer['args']
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    opt_args['weight_decay'],
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                    0.0
            },
        ]
        parameters_count = {
            "no_decay":
                sum(p.numel() for p in optimizer_grouped_parameters[1]['params']),
            "decay":
                sum(p.numel() for p in optimizer_grouped_parameters[0]['params'])
        }
        total_parameters = parameters_count['decay'] + parameters_count['no_decay']
        parameters_count['total'] = total_parameters
    else:
        optimizer_grouped_parameters = list(partition.partition.parameters())
        parameters_count = sum(p.numel() for p in optimizer_grouped_parameters)
    print(f"-I- optimized parameters count: {parameters_count}")
    return optimizer_grouped_parameters


if __name__ == '__main__':
    from types import SimpleNamespace

    args = SimpleNamespace(cpu=False, world_size=8, nnodes=1)
    print(get_rank_to_device_map(args))


class SmallerLastBatchPolicy(Enum):
    ProportionalStep = auto()
    DropReminder = auto()
