import warnings
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

from pipe.models.simple_partitioning_config import PipelineConfig
from .datasets import (
    AVAILABLE_DATASETS,
    # DICT_DATASET_JUST_XY_FUNC,
    HARDCODED_JUST_XY,
    MyNewDistributedSampler
)
from .hardcoded_dirs import DEFAULT_DATA_DIR


# new_distributed_get_train_valid_dl_from_args  (train, valid)
# simplified_get_train_valid_dl_from_args  (train, valid)
# get_separate_just_x_or_y_train_test_dl_from_args  (train, valid)
# get_separate_just_x_or_y_test_dl_from_args: (just the test dataloader)

# NOTE: **kw here are keywords for DataLoader.

###################################
# From args and key words.
###################################


def _is_hardcoded_xy(args):
    is_hardcoded_xy = args.dataset in HARDCODED_JUST_XY
    return is_hardcoded_xy


def get_just(args, pipe_config=None):
    is_hardcoded_xy = _is_hardcoded_xy(args)
    if is_hardcoded_xy:

        if pipe_config is None:  # legacy
            warnings.warn("using hardcoded xy without pipe config (to be deprecated)")
            if args.stage == 0:
                just = 'x'
            elif args.stage == args.num_stages - 1:
                just = 'y'
            else:
                just = None
        else:
            # Hardcoded, but smarter: by depth (allows more flexibility)
            pipe_config: PipelineConfig
            my_depth = pipe_config.get_depth_for_stage(args.stage)
            if my_depth == pipe_config.pipeline_depth - 1:
                just = 'x'
            elif my_depth == 0:
                just = 'y'
            else:
                just = None
    else:
        pipe_config: PipelineConfig
        inputs_from_dl = pipe_config.get_dataset_inputs_for_stage(args.stage)
        just = inputs_from_dl
        print(f"stage{args.stage}: inferred inputs from config: {just}")

    return just


def get_dataloader_keywords(args):
    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    else:
        dl_kw['pin_memory'] = True

    dl_kw['num_workers'] = args.num_data_workers
    dl_kw['drop_last'] = True

    if getattr(args, "dont_drop_last", False):
        dl_kw['drop_last'] = False

    return dl_kw


def get_data_dir(args):
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR
    return DATA_DIR


def get_separate_dls_from_args(args,
                               pipe_config: Optional[PipelineConfig] = None,
                               dataset_keywords: Optional[Dict[str, Any]] = None,
                               verbose=False,
                               # currently unused:
                               shuffle_train=True, ):
    if dataset_keywords is None:
        dataset_keywords = dict()
    just = get_just(args, pipe_config=pipe_config)
    if not just and not getattr(args, "load_extra_inputs", False):  # Empty
        return None, None, [], None
    data_dir = get_data_dir(args)
    dataloader_keywords = get_dataloader_keywords(args)
    assert 'shuffle' not in dataloader_keywords, str(dataloader_keywords)
    experiment_manual_seed = torch.initial_seed()

    try:
        handler = AVAILABLE_DATASETS[args.dataset](just=just, DATA_DIR=data_dir, args=args, **dataset_keywords)
    except KeyError as e:
        print("available datasets", AVAILABLE_DATASETS.keys())
        raise e

    ds_train = handler.get_train_ds(just=just, DATA_DIR=data_dir, args=args, **dataset_keywords)
    ds_test = handler.get_test_ds(just=just, DATA_DIR=data_dir, args=args, **dataset_keywords)
    dataloader_keywords = handler.modify_dataloader_keywords(dataloader_keywords)
    extra = handler.get_modify_trainer_fn()

    # Note: choosing None will infer these args from torch.distributed calls.
    # HACK: we set everything to rank 0 and 1 replica.
    # (we do this to utilize the tested generator code inside the distributed sampler)
    # TODO: replicated stages
    train_sampler = MyNewDistributedSampler(experiment_manual_seed,
                                            ds_train,
                                            num_replicas=1,
                                            rank=0,
                                            shuffle=shuffle_train)

    test_sampler = MyNewDistributedSampler(
        experiment_manual_seed, ds_test, num_replicas=1, rank=0,
        shuffle=False) if ds_test is not None else None

    dl_train = DataLoader(
        ds_train,
        args.bs_train,
        shuffle=False,
        sampler=train_sampler,
        **dataloader_keywords)
    dl_test = DataLoader(
        ds_test,
        args.bs_test,
        shuffle=False,
        sampler=test_sampler,
        **dataloader_keywords) if ds_test is not None else None

    if verbose:
        n_samples_train = len(dl_train) * args.bs_train
        n_samples_test = len(dl_test) * args.bs_test if dl_test is not None else 0
        print(f'Train: {n_samples_train} samples')
        print(f'Test: {n_samples_test} samples')

    if extra:
        if isinstance(extra, list):
            assert len(extra) == 1
            extra = extra[0]

    return dl_train, dl_test, list(filter(
        None, [train_sampler, test_sampler])), extra


def add_dataset_argument(parser, default='cifar10', required=False):
    parser.add_argument('--dataset',
                        default=default,
                        choices=list(AVAILABLE_DATASETS.keys()),
                        required=required)
