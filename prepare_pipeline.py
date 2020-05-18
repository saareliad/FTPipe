import torch

from misc.filelogger import FileLogger
from pipeline import dp_sim
from datasets import (simplified_get_train_valid_dl_from_args,
                      get_separate_just_x_or_y_train_test_dl_from_args,
                      get_separate_just_x_or_y_test_dl_from_args)

import optimizers.lr_scheduler
from pipeline.work_schedulers import AVAILABLE_WORK_SCHEDULERS
from pipeline.weight_stashing import WeightStasher
from pipeline import TrueWeightsStorage

import models
from pipeline import CommunicationHandlerBase, get_auto_comm_handler_cls
from pipeline.communication.multiprocessing import MultiprocessingCommunicationHandler

from pipeline.communication.multiprocessing_pull import MultiprocessingCommunicationHandler as PullMultiprocessingCommunicationHandler

from pipeline.partition_manager import SinglePartitionManager
from pipeline.mp_partition_manager import SinglePartitionManager as MPSinglePartitionManager

from pipeline.partition_manager import GPipePartitionManager

from pipeline.training import AVAILABLE_TRAINERS
from pipeline.tasks import AVAILABLE_TASKS
from pipeline.stats import get_statistics  # , Stats
from pipeline.weight_prediction import get_sched_predictor

from pipeline.weight_prediction import get_weight_predictor as get_weight_predictor_partial

from pipeline.gap_aware import (get_sgd_gap_aware_cls, get_adam_gap_aware_cls,
                                get_adamw_gap_aware_cls)
from optimizers import AVAILBALE_OPTIMIZERS

from models import parse_config
from datasets.lm import lm_collate_factory


def auto_file_name(args):
    assert hasattr(args, "auto_file_name")
    wp = args.weight_prediction['type'] if hasattr(
        args, "weight_prediction") else 'stale'
    ws = "ws_" if getattr(args, "weight_stashing", False) else ""
    ga = "ga_" if hasattr(args, "gap_aware") else ""
    bs = f"bs_{args.bs_train * args.step_every}"
    se = f"se_{args.step_every}"
    ga_just_for_loss = "gaJFL_" if getattr(args, 'gap_aware_just_loss',
                                           False) else ""

    if 'gpipe' == args.work_scheduler.lower():
        s = f'{args.model}_{args.dataset}_gpipe_{bs}_{se}_seed_{args.seed}'
    else:
        s = f'{args.model}_{args.dataset}_{wp}_{ws}{ga}{bs}_{se}_{ga_just_for_loss}seed_{args.seed}'
    args.out_filename = f"{args.out_filename}_{s}"
    print(f"Out File Name will be: {args.out_filename}")


def create_comm_handler(args, comm_init_args,
                        device) -> CommunicationHandlerBase:

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
    if hasattr(args, "lr_scheduler"):
        # should_step = False
        # TODO: auto-calculate numbers like num_training_steps
        attr = getattr(args, 'lr_scheduler')

        preproc_args = attr.get("preproc_args", None)
        if preproc_args:
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
                else:
                    raise NotImplementedError(
                        f"Unsupported preprocess argument {preproc_command}")

        if attr['type'] in optimizers.lr_scheduler.AVAILABLE_LR_SCHEDULERS:
            scheduler_cls = getattr(optimizers.lr_scheduler, attr['type'])
            # should_step = True
        else:
            scheduler_cls = getattr(torch.optim.lr_scheduler, attr['type'])

        scheduler = scheduler_cls(optimizer, **attr['args'])

        # TODO: also get scheduler for sched aware prediction.
        sched_aware_stuff = (scheduler_cls, attr['args'])

        # TODO: in some optimizers version we can bendfit from lr=0 (build momentum in place)
        # while on others we don't, and better step.
        # For now I just leave it as is.
        # OPTIONAL: Perform a dummy step to avoid lr=0 at the start of the training.
        # if should_step:
        #     scheduler.step()
        return scheduler, sched_aware_stuff


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
                delattr(args, 'weight_prediction')
                # Return, so we can use it for stuff like file nameing, etc.
                return res


def get_weight_predictor(args,
                         optimizer,
                         scheduler=None,
                         true_weights_storage=None,
                         sched_aware_stuff=None):
    """
        Returns:
            weight_predictor,
            nag_with_predictor: bool
    """
    assert (true_weights_storage is not None
            )  # TODO: this should be normal argument when its stable
    if not hasattr(args, 'weight_prediction'):
        return None, None

    optimizer_type = getattr(args, 'optimizer')['type']
    pred = getattr(args, 'weight_prediction')
    pred_mem = pred['args']['pred_mem']
    nag_with_predictor = pred['args'].get('nag_with_predictor', False)

    assert (pred_mem in {"clone", "calc"})
    assert (pred['type'] == "msnag")
    assert ('sgd' in optimizer_type or 'adam' in optimizer_type)

    sched_predictor = None
    # If we have sched aware:
    if pred['args'].get("sched_aware", False):
        print("-I- using sched aware weight prediction")
        assert scheduler is not None
        assert sched_aware_stuff is not None
        scheduler_class = sched_aware_stuff[0]
        scheduler_kw = sched_aware_stuff[1]
        sched_predictor = get_sched_predictor(optimizer, scheduler_class,
                                              **scheduler_kw)
        sched_predictor.patch_scheduler(scheduler)
        assert 'adam' in optimizer_type  # Remove after we implement for sgd.

    weight_predictor = get_weight_predictor_partial(
        optimizer_type,
        pred_mem,
        optimizer,
        scheduler=sched_predictor,
        nag_with_predictor=nag_with_predictor,
        true_weights_storage=true_weights_storage,
        sched_predictor=sched_predictor)

    return weight_predictor, nag_with_predictor


def get_dataloaders(args, explicit_separated_dataset=False, **kw):
    # TODO: currently assuming that only 1 rank is x or y.
    # will have to fix this for replicated.
    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    else:
        dl_kw['pin_memory'] = True

    dl_kw['num_workers'] = args.num_data_workers
    dl_kw['drop_last'] = True

    if getattr(args, "dont_drop_last", False):
        dl_kw['drop_last'] = False

    if "lm" in args.task:
        # FIXME
        # NOTE: From the function get_wikitext2_raw_train_valid_ds
        tokenizer = kw.pop('tokenizer')
        overwrite_cache = getattr(args, 'overwrite_cache', False)
        dataset_keywords = dict(model_name_or_path=args.model_name_or_path,
                                tokenizer=tokenizer,
                                train_seq_len=args.train_seq_len,
                                test_seq_len=args.test_seq_len,
                                overwrite_cache=overwrite_cache)
        collate = lm_collate_factory(tokenizer)
        dl_kw['collate_fn'] = collate
    elif 'squad' in args.task:
        tokenizer = kw.pop('tokenizer')
        overwrite_cache = getattr(args, 'overwrite_cache', False)
        dataset_keywords = dict(
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            threads=args.threads,
            version_2_with_negative=args.task == 'squad2',
            save=True,  # TODO: according to Ranks for stage replication
            # NOTE: deleted
            # train_seq_len=args.train_seq_len,
            # test_seq_len=args.test_seq_len,
            overwrite_cache=overwrite_cache)
    else:
        dataset_keywords = {}

    if explicit_separated_dataset:
        train_dl, test_dl, samplers = get_separate_just_x_or_y_train_test_dl_from_args(
            args, verbose=False, dataset_keywords=dataset_keywords, **dl_kw)
    else:
        # Note: sometimes used to infer all parameters, (by all partitions).
        train_dl, test_dl, *samplers = simplified_get_train_valid_dl_from_args(
            args, verbose=False, dataset_keywords=dataset_keywords, **dl_kw)

    return train_dl, test_dl, samplers


def get_just_test_dataloader(args, explicit_separated_dataset=False, **kw):
    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    else:
        dl_kw['pin_memory'] = True

    dl_kw['num_workers'] = args.num_data_workers
    dl_kw['drop_last'] = True
    if getattr(args, "dont_drop_last", False):
        dl_kw['drop_last'] = False

    if "lm" in args.task:
        # NOTE: From the function get_wikitext2_raw_test_ds
        tokenizer = kw.pop('tokenizer')
        overwrite_cache = getattr(args, 'overwrite_cache', False)
        dataset_keywords = dict(model_name_or_path=args.model_name_or_path,
                                tokenizer=tokenizer,
                                test_seq_len=args.test_seq_len,
                                overwrite_cache=overwrite_cache)
        collate = lm_collate_factory(tokenizer)
        dl_kw['collate_fn'] = collate
    elif 'squad' in args.task:
        raise NotImplementedError()
    else:
        dataset_keywords = {}

    if explicit_separated_dataset:
        test_dl, sampler = get_separate_just_x_or_y_test_dl_from_args(
            args,
            verbose=False,
            test_dataset_keywords=dataset_keywords,
            **dl_kw)
    else:
        # Note: sometimes used to infer all parameters, (by all partitions).
        raise NotImplementedError()
        # test_dl, sampler = simplified_get_test_dl_from_args(
        #     args, verbose=False, dataset_keywords=dataset_keywords, **dl_kw)

    return test_dl, sampler


def get_device(args, local_rank):
    if hasattr(args, "stage_to_device_map"):
        stage_to_device_map = args.stage_to_device_map
        cuda_device_id = stage_to_device_map[local_rank]
        device = torch.device('cpu' if args.cpu else f"cuda:{cuda_device_id}")
    else:
        device = torch.device('cpu' if args.cpu else f"cuda:{local_rank}")
    return device


def get_rank_to_device_map(args):
    return {rank: get_device(args, local_rank=rank) for rank in range(args.world_size)}


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
    return optimizer_cls


def get_optimizer(args, optimizer_cls, parameters):
    # without the list, python 3.8 pytorch 1.5: TypeError: object of type 'generator' has no len()
    parameters = list(parameters)
    if len(parameters) == 0:
        if not getattr(args, "allow_stateless", False):
            raise ValueError(f"Got stateless partition {args.stage}")

    optimizer = optimizer_cls(parameters, **args.optimizer['args'])

    return optimizer


def prepare_pipeline(args, shared_ctx=None, COMM_VERSION=1):

    is_gpipe = "gpipe" == args.work_scheduler.lower()
    if not args.is_multiprocessing_worker:
        # select a partition manager
        if is_gpipe:
            print("Preparing pipeline for GPipe")
            partition_cls = GPipePartitionManager
        else:
            partition_cls = SinglePartitionManager
    else:
        # Partition manger for multiprocessing
        partition_cls = MPSinglePartitionManager
        COMM_VERSION = 2
        if is_gpipe:
            raise NotImplementedError()

    # get work scheduler
    work_scheduler = AVAILABLE_WORK_SCHEDULERS.get(args.work_scheduler)

    # set device
    local_rank_to_device_map = get_rank_to_device_map(args)
    device = local_rank_to_device_map[args.local_rank]
    if not args.cpu:
        torch.cuda.set_device(device)

    # Parse partitioning config and requires args
    print(f"Loading partitioned model and dataset...")
    model_instance = None
    dataset_keywords = {}
    if args.model in models.transformers_cfg.MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS.keys(
    ):
        model_instance, tokenizer, config = models.transformers_utils.get_model_tokenizer_and_config_by_name(
            args.model)
        dataset_keywords['tokenizer'] = tokenizer

    parsed_config = parse_config.PartitioningConfigParser(
        args.model,
        args.rank,
        args.bs_train,
        args.bs_test,  # NOTE: changed name
        model_instance=model_instance,
        send_target_in_pipe=not ("_sep" in args.task))

    comm_init_args = parsed_config.comm_init_args()

    training_tensor_dtypes = parsed_config.training_tensor_dtypes
    eval_tensor_shapes = parsed_config.eval_tensor_shapes
    training_tensor_shapes = parsed_config.training_tensor_shapes

    args.num_stages = parsed_config.num_stages
    args.stage = parsed_config.stage
    model = parsed_config.model

    is_first_partition = args.stage == 0
    is_last_partition = args.stage == args.num_stages - 1

    assert (args.epochs >= 1 or args.steps >= 1)
    assert (not (args.stage is None))

    logger = FileLogger(args.logdir,
                        global_rank=args.rank,
                        local_rank=args.local_rank,
                        name='msnag',
                        world_size=args.world_size,
                        name_prefix=args.out_filename)  # FIXME: real name

    # Comm handler
    if COMM_VERSION == 1:
        comm_handler = create_comm_handler(args, comm_init_args, device)
        comm_handler.init_process_group()
    elif COMM_VERSION == 2:
        # Multiprocessing
        stage_to_device_map = []  # TODO
        v2_args = (shared_ctx, stage_to_device_map, local_rank_to_device_map)
        comm_handler = create_comm_handler_v2(args, comm_init_args, device, v2_args)
    else:
        raise NotImplementedError("In progress")

    # Try getting separate X,Y dataloaders
    if is_first_partition or is_last_partition:
        explicit_separated_dataset = "_sep" in args.task

        train_dl, test_dl, samplers = get_dataloaders(
            args,
            explicit_separated_dataset=explicit_separated_dataset,
            **dataset_keywords)
    else:
        train_dl, test_dl, samplers = None, None, []
    del dataset_keywords

    # instead of loading dl on every device,
    # when not needed - can just send the length as a message
    if args.rank == 0:
        assert is_first_partition
        train_dl_len, test_dl_len = len(train_dl), len(test_dl)
        logger.info(f"train_dl_len {train_dl_len}")
        logger.info(f"test_dl_len {test_dl_len}")

        train_dataset_len, test_dataset_len = len(train_dl.dataset), len(
            test_dl.dataset)
        logger.info(f"train_dataset_len {train_dataset_len}")
        logger.info(f"test_dataset_len {test_dataset_len}")

        # TODO: also support replicated
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

    ##############################
    # Until here its common,
    # To GPipe too.
    ##############################

    # Will automtomatically change trainer to gap aware compatible trainer
    partition_using_gap_aware = False
    if not is_gpipe:
        partition_using_gap_aware = hack_trainer_type_to_gap_aware(args)

    if partition_using_gap_aware:
        logger.info(f"Stage {args.stage} will use Gap Aware")

    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer['type'])
    task_cls = AVAILABLE_TASKS.get(args.task)
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
        training_tensor_shapes,
        eval_tensor_shapes,
        training_tensor_dtypes,
        training_tensor_dtypes,  # HACK, FIXME
        device,
        is_last_partition,
        is_first_partition,
        log_frequency=args.log_frequency,
        max_buffers=args.max_buffers,
        step_every=args.step_every,
        keep_buffers_alive=args.keep_buffers_alive,
        use_recomputation=(not args.no_recomputation),
        gap_aware_just_loss=gap_aware_just_loss,
        use_pre_loaded_label_input=getattr(args, "use_pre_loaded_label_input",
                                           False),
        weight_stashing_just_for_stats=getattr(
            args, "weight_stashing_just_for_stats", False),
        stateless_tied=getattr(args, "stateless_tied", False),
        last_batch_train_shapes=last_batch_train_shapes,
        last_batch_test_shapes=last_batch_test_shapes)

    # support for simulating stage replication (dev)
    if hasattr(args, "ddp_sim_num_gpus") and args.ddp_sim_num_gpus > 1:
        print(
            f"-I- simulating DDP accuracy with {args.ddp_sim_num_gpus} (DDP) GPUs per stage"
        )
        dp_sim.convert_to_num_gpus(partition.partition, args.ddp_sim_num_gpus)

    # After the partition is on its device:
    # Set optimizer
    # If is transformer, use grouped parameters.
    if 'lm' in args.task or 'squad' in args.task:
        # No weight decay for some parameters.
        model = partition.partition
        # NOTE: it works even if len(model.paramerters()) == 0
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
        lengths = {
            "no_decay": len(optimizer_grouped_parameters[1]['params']),
            "decay": len(optimizer_grouped_parameters[0]['params'])
        }
        # total_length = lengths['decay'] + lengths['no_decay']
        print(f"-I- optimizer_grouped_parameters: {lengths}")
    else:
        optimizer_grouped_parameters = partition.partition.parameters()

    # if we replace wp with nesterov, we save the wp arg, and set it back for config and auto experiment naming.
    stashed_wp_arg = None
    if not is_gpipe:
        if is_last_partition:
            stashed_wp_arg = try_replace_prediction_with_nesterov(args)

    optimizer = get_optimizer(args, optimizer_cls,
                              optimizer_grouped_parameters)

    if not is_gpipe:
        true_weights_storage = TrueWeightsStorage(optimizer)
        partition.set_true_weights_storage(true_weights_storage)

    if args.flush_rate > 0 and args.flush_rate < args.step_every:
        raise NotImplementedError()

    # Set Scheduler
    # TODO: scheduler for sched aware prediction
    scheduler, sched_aware_stuff = get_lr_scheduler(args, optimizer)

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

    partition.set_trainer(trainer)
    partition.set_lr_scheduler(scheduler)

    if not is_gpipe:
        # Set Weight predictor
        weight_predictor, nag_with_predictor = get_weight_predictor(
            args,
            optimizer,
            scheduler=scheduler,
            true_weights_storage=true_weights_storage,
            sched_aware_stuff=sched_aware_stuff,
        )
        if weight_predictor:
            partition.set_weight_predictor(weight_predictor,
                                           nag_with_predictor)

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

    # Set Task
    task = task_cls(device, is_last_partition, is_first_partition)
    partition.set_task(task)

    if hasattr(args, "auto_file_name"):
        # make sure this specific replacement does not ruin experiment name
        if is_last_partition and stashed_wp_arg:
            args.weight_prediction = stashed_wp_arg

        auto_file_name(args)

        if is_last_partition and stashed_wp_arg:
            del args.weight_prediction
            del stashed_wp_arg

    return (logger, train_dl, test_dl, is_first_partition, is_last_partition,
            partition, statistics, train_dl_len, test_dl_len, samplers)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentsParser()

    args = parser.parse_args()

    (logger, train_dl, test_dl, is_first_partition, is_last_partition,
     partition, statistics, train_dl_len, test_dl_len,
     samplers) = prepare_pipeline(args)
