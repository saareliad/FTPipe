import torch

from experiments.experiments import ArgsStasher, auto_file_name
from misc.filelogger import FileLogger
from data import get_separate_dls_from_args

import models
from models import parse_config

from pipeline.partition_manager import (SinglePartitionManager, GPipePartitionManager)
from pipeline import CommunicationHandlerBase, get_auto_comm_handler_cls
from pipeline.communication.multiprocessing import MultiprocessingCommunicationHandler
from pipeline.statistics import get_statistics  # , Stats
from pipeline.weight_prediction import get_sched_predictor
from pipeline.weight_prediction import get_weight_predictor as get_weight_predictor_partial
from pipeline.gap_aware import (get_sgd_gap_aware_cls, get_adam_gap_aware_cls,
                                get_adamw_gap_aware_cls)
from pipeline.weight_stashing import WeightStasher
from pipeline import TrueWeightsStorage
from pipeline import dp_sim

import optimizers.lr_scheduler

# TODO: migrate to `register_xxx()` convention
from pipeline.work_schedulers import AVAILABLE_WORK_SCHEDULERS
from pipeline.training import AVAILABLE_TRAINERS
from pipeline.data_propagation import AVAILABLE_PROPAGATORS
from optimizers import AVAILBALE_OPTIMIZERS
# from data import AVAILABLE_DATASETS


def get_propagator_cls(args):
    propagator_cls = AVAILABLE_PROPAGATORS.get(args.data_propagator)
    return propagator_cls


def get_trainer_cls(args):
    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer['type'])
    return trainer_cls


def is_huggingface_transformer(args):
    return args.model in models.transformers_cfg.MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS.keys()


def is_explicit_non_seperated_dataset(args):
    return "_nonsep" in args.data_propagator


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


def get_dataloaders(args,
                    pipe_config=None,
                    dataset_keywords=dict()):
    # TODO: replicated
    if not is_explicit_non_seperated_dataset(args):
        train_dl, test_dl, samplers, extra = get_separate_dls_from_args(
            args,
            pipe_config=pipe_config,
            verbose=False,
            dataset_keywords=dataset_keywords,
        )
    else:
        raise NotImplementedError("now deprecated")
    return train_dl, test_dl, samplers, extra


def get_device(args, local_rank):
    if hasattr(args, "stage_to_device_map"):
        stage_to_device_map = args.stage_to_device_map
        cuda_device_id = stage_to_device_map[local_rank]
        device = torch.device('cpu' if args.cpu else f"cuda:{cuda_device_id}")
    else:
        device = torch.device('cpu' if args.cpu else f"cuda:{local_rank}")
    return device


def get_rank_to_device_map(args):
    return {
        rank: get_device(args, local_rank=rank)
        for rank in range(args.world_size)
    }


def hack_trainer_type_to_gap_aware(args):
    def hack():
        args.trainer['type'] += "_gap_aware"

    if hasattr(args, 'gap_aware'):
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
        elif args.gap_aware['policy'] == 'all_except_last_two':
            is_last_partition = args.local_rank == args.world_size - 1
            is_almost_last_partition = args.local_rank == args.world_size - 2
            if (not is_last_partition) and (not is_almost_last_partition):
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
    model_instance = None
    dataset_keywords = {}
    if is_huggingface_transformer(args):
        if cache is None:
            handler = models.AVAILABLE_MODELS.get(args.model)
            model_instance = handler.get_normal_model_instance()
            tokenizer = handler.tokenizer
            config = handler.config

            if save_cache:
                cache = (model_instance, tokenizer, config)
        else:
            model_instance, tokenizer, config = cache

        dataset_keywords['tokenizer'] = tokenizer
        dataset_keywords['config'] = config

    parsed_config = parse_config.PartitioningConfigParser(
        args.model,
        args.rank,
        args.bs_train,
        args.bs_test,  # NOTE: changed name
        model_instance=model_instance,
        send_target_in_pipe=("_nonsep" in args.data_propagator),
        prefer_seq_sends=getattr(args, "prefer_seq_sends",True))

    pipe_config = parsed_config.pipe_config
    args.num_stages = parsed_config.num_stages
    args.stage = parsed_config.stage
    train_dl, test_dl, samplers, extra = get_dataloaders(
        args,
        pipe_config=pipe_config,
        dataset_keywords=dataset_keywords)

    return cache


def prepare_pipeline(args, shared_ctx=None, COMM_VERSION=1):
    is_gpipe = "gpipe" == args.work_scheduler.lower()
    # select a partition manager
    if is_gpipe:
        print("Preparing pipeline for GPipe")
        partition_cls = GPipePartitionManager
    else:
        partition_cls = SinglePartitionManager

    if args.is_multiprocessing_worker:
        # multiprocessing communication handler
        COMM_VERSION = 2

    # get work scheduler
    work_scheduler = AVAILABLE_WORK_SCHEDULERS.get(args.work_scheduler)

    # set device
    local_rank_to_device_map = get_rank_to_device_map(args)
    device = local_rank_to_device_map[args.local_rank]
    if not args.cpu:
        torch.cuda.set_device(device)

    # Parse partitioning config and requires args
    # TODO: some easier way to get original model and the config used during partitioning (WIP)
    print(f"Loading partitioned model and dataset...")
    model_instance = None
    dataset_keywords = {}
    if is_huggingface_transformer(args):
        handler = models.AVAILABLE_MODELS.get(args.model)
        model_instance = handler.get_normal_model_instance()
        tokenizer = handler.tokenizer
        config = handler.config
        del handler.config
        del handler.tokenizer

        dataset_keywords['tokenizer'] = tokenizer
        dataset_keywords['config'] = config

    parsed_config = parse_config.PartitioningConfigParser(
        args.model,
        args.rank,
        args.bs_train,
        args.bs_test,  # NOTE: changed name
        model_instance=model_instance,
        send_target_in_pipe=("_nonsep" in args.data_propagator),
        prefer_seq_sends=getattr(args, "prefer_seq_sends",True))

    pipe_config = parsed_config.pipe_config

    comm_init_args = parsed_config.comm_init_args()

    training_tensor_dtypes = parsed_config.training_tensor_dtypes
    eval_tensor_shapes = parsed_config.eval_tensor_shapes
    training_tensor_shapes = parsed_config.training_tensor_shapes

    args.num_stages = parsed_config.num_stages
    args.stage = parsed_config.stage

    # NOTE: here its the sliced model.
    model = parsed_config.model
    # del parsed_config.model  # NOTE: can delete the extra reference to possibly save mem.
    del model_instance
    del handler.normal_model_instance

    model.device = device

    is_first_partition = args.stage == 0
    is_last_partition = args.stage == args.num_stages - 1

    assert (args.epochs >= 1 or args.steps >= 1)
    assert (not (args.stage is None))
    # FIXME: real name
    logger = FileLogger(args.logdir,
                        global_rank=args.rank,
                        local_rank=args.local_rank,
                        name='msnag',
                        world_size=args.world_size,
                        name_prefix=args.out_filename)

    eval_tensor_dtypes = training_tensor_dtypes  # HACK, TODO

    # Get dataloaders needed for this stage
    train_dl, test_dl, samplers, extra = get_dataloaders(
        args,
        pipe_config=pipe_config,
        dataset_keywords=dataset_keywords)

    del dataset_keywords

    # Comm handler
    if COMM_VERSION == 1:
        comm_handler = create_comm_handler(args, comm_init_args, device)
        comm_handler.init_process_group()
    elif COMM_VERSION == 2:
        # Multiprocessing
        stage_to_device_map = []  # TODO
        v2_args = (shared_ctx, stage_to_device_map, local_rank_to_device_map)
        comm_handler = create_comm_handler_v2(args, comm_init_args, device,
                                              v2_args)
    else:
        raise NotImplementedError("In progress")

    # instead of loading dl on every device just to get its length
    # we synchronize length as a message, from first stage
    (last_batch_diff_test,
     last_batch_diff_train,
     test_dl_len,
     train_dl_len) = synchronize_dataloaders_length(args,
                                                    is_first_partition,
                                                    logger,
                                                    test_dl,
                                                    train_dl)
    if last_batch_diff_train > 0:
        last_batch_train_shapes = parsed_config.get_shapes(
            last_batch_diff_train)
    else:
        last_batch_train_shapes = None

    if last_batch_diff_test > 0:
        last_batch_test_shapes = parsed_config.get_shapes(last_batch_diff_test)
    else:
        last_batch_test_shapes = None

    # Get expected training steps:
    if args.epochs > 0 and args.steps < 0:
        steps_per_epoch = train_dl_len // args.step_every
        if train_dl_len % args.step_every > 0:
            steps_per_epoch += 1
        expected_training_steps = steps_per_epoch * args.epochs
    else:
        raise NotImplementedError()

    args.steps_per_epoch = steps_per_epoch
    args.expected_training_steps = expected_training_steps

    buffers_ctx = (
        training_tensor_shapes,
        eval_tensor_shapes,
        training_tensor_dtypes,
        eval_tensor_dtypes,
        last_batch_train_shapes,
        last_batch_test_shapes,
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
        partition_using_gap_aware = hack_trainer_type_to_gap_aware(args)

    if partition_using_gap_aware:
        logger.info(f"Stage {args.stage} will use Gap Aware")

    trainer_cls = get_trainer_cls(args)
    propagator_cls = get_propagator_cls(args)
    optimizer_cls = get_optimizer_cls(args, partition_using_gap_aware)
    statistics = get_statistics(args.statistics,
                                is_last_partition=is_last_partition)
    assert not (statistics is None)

    # Gap aware penatly just for the loss
    gap_aware_just_loss = False
    if not is_gpipe:
        gap_aware_just_loss = getattr(args, 'gap_aware_just_loss', False)
        if gap_aware_just_loss:
            if is_last_partition:
                gap_aware_just_loss = False
            else:
                if args.no_recomputation:
                    raise NotImplementedError(
                        "gap_aware_just_loss works only with recomputation on")

    # Init the partition manager itself, warping the model and loading it to device.
    partition = partition_cls(
        args.stage,
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
        stateless_tied=getattr(args, "stateless_tied", False),
        is_mp=args.is_multiprocessing_worker,
        req_grad=parsed_config.req_grad,
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
    if not is_gpipe and is_last_partition:
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
            if not is_last_partition:

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

    return (logger, train_dl, test_dl, is_first_partition, is_last_partition,
            partition, statistics, train_dl_len, test_dl_len, samplers)


def synchronize_dataloaders_length(args, is_first_partition, logger, test_dl, train_dl):
    if args.rank == 0:
        assert is_first_partition
        train_dl_len, test_dl_len = len(train_dl), len(test_dl)
        logger.info(f"train_dl_len {train_dl_len}")
        logger.info(f"test_dl_len {test_dl_len}")

        train_dataset_len, test_dataset_len = len(train_dl.dataset), len(
            test_dl.dataset)
        logger.info(f"train_dataset_len {train_dataset_len}")
        logger.info(f"test_dataset_len {test_dataset_len}")

        # TODO: support replicated
        last_batch_diff_train = train_dataset_len % args.bs_train if not train_dl.drop_last else 0
        last_batch_diff_test = test_dataset_len % args.bs_test if not test_dl.drop_last else 0

        data = [
            train_dl_len, test_dl_len, last_batch_diff_train,
            last_batch_diff_test
        ]
        data = torch.tensor(data, dtype=torch.long)
    else:
        data = torch.zeros(4, dtype=torch.long)
    torch.distributed.broadcast(data, 0)
    train_dl_len = data[0].item()
    test_dl_len = data[1].item()
    last_batch_diff_train = data[2].item()
    last_batch_diff_test = data[3].item()
    return last_batch_diff_test, last_batch_diff_train, test_dl_len, train_dl_len


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentsParser()

    args = parser.parse_args()

    (logger, train_dl, test_dl, is_first_partition, is_last_partition,
     partition, statistics, train_dl_len, test_dl_len,
     samplers) = prepare_pipeline(args)
