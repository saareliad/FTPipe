import os
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, Sampler
from typing import List, Tuple

from .hardcoded_dirs import DEFAULT_DATA_DIR


from .datasets import (
    AVAILABLE_DATASETS,
    # DICT_DATASET_JUST_XY_FUNC,
    MyNewDistributedSampler
)

# new_distributed_get_train_valid_dl_from_args  (train, valid)
# simplified_get_train_valid_dl_from_args  (train, valid)
# get_separate_just_x_or_y_train_test_dl_from_args  (train, valid)
# get_separate_just_x_or_y_test_dl_from_args: (just the test dataloader)

# NOTE: **kw here are keywords for DataLoader.

###################################
# From args and key words.
###################################


def _is_hardcoded_xy(args):
    HARDCODED_JUST_XY = {"lm", "cv"}  # HACK: used to hardcode this.
    # TODO: it should be by datasets actually and not task, will handle it later
    is_hardcoded_xy = args.task in HARDCODED_JUST_XY
    return is_hardcoded_xy


def get_just(args, pipe_config=None):
    is_hardcoded_xy = _is_hardcoded_xy(args)
    if is_hardcoded_xy or pipe_config is None:  # legacy
        print("-I- using hardcoded xy (to be deprecated)")
        if args.stage == 0:
            just = 'x'
        elif args.stage == args.num_stages - 1:
            just = 'y'
        else:
            just = None
    else:
        pcs = pipe_config.stages[args.stage]
        inputs_from_dl = [
            i for i in pcs.inputs if i in pipe_config.model_inputs
        ]
        just = inputs_from_dl
        print(f"stage{args.stage}: infered inputs from config: {just}")

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
    # TODO: according to ranks, for replicated stages.
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR
    return DATA_DIR


# TODO: handle dl_kw being propegated
def get_separate_dls_from_args(args,
                               pipe_config=None,
                               dataset_keywords=dict(),
                               verbose=False,
                               # currently unused:
                               shuffle_train=True,):
    just = get_just(args, pipe_config=pipe_config)
    if not just and not getattr(args, "load_extra_inputs", False):  # Empty
        return None, None, [], None
    data_dir = get_data_dir(args)
    dataloader_keywords = get_dataloader_keywords(args)
    assert 'shuffle' not in dataloader_keywords, str(dataloader_keywords)
    experiment_manual_seed = torch.initial_seed()

    handler = AVAILABLE_DATASETS[args.dataset](just=just, DATA_DIR=data_dir, args=args, **dataset_keywords)
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
        assert len(extra) == 1
        extra = extra[0]

    return dl_train, dl_test, list(filter(
        None, [train_sampler, test_sampler])), extra


def add_dataset_argument(parser, default='cifar10', required=False):
    parser.add_argument('--dataset',
                        default=default,
                        choices=list(AVAILABLE_DATASETS.keys()),
                        required=required)
