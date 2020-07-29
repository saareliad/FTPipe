import os
import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, Sampler
from typing import List, Tuple

# Datasets
from .cv import get_cifar_100_train_test_ds, get_imagenet_train_test_ds, get_cifar_10_train_test_ds

from .lm import (get_wikitext2_raw_train_valid_test_ds,
                 get_wikitext2_raw_train_test_ds,
                 get_wikitext2_raw_train_valid_ds, get_wikitext2_raw_test_ds)

# Just x or y Datasets
from .cv import (get_cifar_10_just_x_or_y_train_test_ds,
                 get_imagenet_just_x_or_y_train_test_ds,
                 get_cifar_100_just_x_or_y_train_test_ds)

from .lm import (get_wt2_just_x_or_y_train_valid_ds,
                 get_wt2_just_x_or_y_train_test_ds,
                 get_wt2_just_x_or_y_test_ds)

# TODO: train and dev, currently dev is None.
from .squad import get_just_x_or_y_train_dev_dataset as get_just_x_or_y_train_dev_dataset_squad
from .glue import get_just_x_or_y_train_dev_dataset as get_just_x_or_y_train_dev_dataset_glue
from .t5_squad import get_just_x_or_y_train_dev_dataset as get_just_x_or_y_train_dev_dataset_t5_squad

# Dataloaders
from .cv import get_cv_train_test_dl
from .lm import get_lm_train_dl, get_lm_eval_dl, get_lm_train_valid_dl

from .hardcoded_dirs import DEFAULT_DATA_DIR

# new_distributed_get_train_valid_dl_from_args  (train, valid)
# simplified_get_train_valid_dl_from_args  (train, valid)
# get_separate_just_x_or_y_train_test_dl_from_args  (train, valid)
# get_separate_just_x_or_y_test_dl_from_args: (just the test dataloader)

AVAILABLE_DATASETS = {
    'cifar10', 'cifar100', 'imagenet', 'wt2', 'squad1', 'squad2', 'glue', "t5_squad"
}

# NOTE: these are functions which returns train and test/validation datasets.
DATASET_TO_DS_FN = {
    'cifar10': get_cifar_10_train_test_ds,
    'cifar100': get_cifar_100_train_test_ds,
    'imagenet': get_imagenet_train_test_ds,
    'wt2': get_wikitext2_raw_train_test_ds,  # TODO
    # 'wt2': get_wikitext2_raw_train_valid_ds,  # TODO
}

# NOTE: functions which returns 2 dataloaders, train and valid/test
# FIXME: DEPRECATED (x,y are not seperate here)
DATASET_TO_DL_FN = {
    'cifar10': get_cv_train_test_dl,
    'cifar100': get_cv_train_test_dl,
    'imagenet': get_cv_train_test_dl,
    'wt2': get_lm_train_valid_dl,  # HACK
}

############################
# Generic "get" functions
############################


def get_train_test_ds(dataset, DATA_DIR=DEFAULT_DATA_DIR, **kw):
    get_dataset_fn = DATASET_TO_DS_FN.get(dataset, None)
    if get_dataset_fn:
        return get_dataset_fn(DATA_DIR=DATA_DIR, **kw)
    else:
        raise ValueError(dataset)


def get_train_test_dl(dataset, *args, **kw):
    get_dl_fn = DATASET_TO_DL_FN.get(dataset, None)
    if get_dl_fn:
        return get_dl_fn(*args, **kw)
    else:
        raise ValueError(dataset)


############################
# Simplified. dataset by name.
############################


# TODO : this works just for CV, for LM
def simplified_get_train_test_dl(dataset,
                                 bs_train,
                                 bs_test,
                                 shuffle_train=True,
                                 verbose=True,
                                 DATA_DIR=DEFAULT_DATA_DIR,
                                 dataset_keywords=dict(),
                                 **kw):
    ds_train, ds_test = get_train_test_ds(dataset,
                                          DATA_DIR=DATA_DIR,
                                          **dataset_keywords)
    # HACK: all as keywords
    dl_train, dl_test = get_train_test_dl(dataset,
                                          ds_train=ds_train,
                                          ds_test=ds_test,
                                          bs_train=bs_train,
                                          bs_test=bs_test,
                                          shuffle=shuffle_train,
                                          **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test


##########################################################
# Distributed. (v2): Using a modified DistributedSampler
###########################################################


class MyNewDistributedSampler(DistributedSampler):
    # Better use this class, as it was tested by pytorch.
    # only problem with it is *deterministic shuffling*, which will be the same for all experiments.
    # so we add experiment seed to make it fun.

    MAX_INT = 2**32  # Used to prevent overflow

    def __init__(self, experiment_manual_seed, *args, **kw):
        super().__init__(*args, **kw)
        self.experiment_manual_seed = experiment_manual_seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        # My only change
        g.manual_seed(
            ((1 + self.epoch) * self.experiment_manual_seed) % self.MAX_INT)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def new_distributed_simplified_get_train_test_dl(dataset,
                                                 bs_train,
                                                 bs_test,
                                                 shuffle_train=True,
                                                 verbose=True,
                                                 DATA_DIR=DEFAULT_DATA_DIR,
                                                 pin_memory=True,
                                                 dataset_keywords=dict(),
                                                 **kw):
    """ Requires:
         that a manual seed is set to the experiment and restorable via torch.initial_seed() """

    ds_train, ds_test = get_train_test_ds(dataset,
                                          DATA_DIR=DATA_DIR,
                                          **dataset_keywords)
    experiment_manual_seed = torch.initial_seed()

    # Note: choosing None will infer these args from torch.distributed calls.
    train_sampler = MyNewDistributedSampler(experiment_manual_seed,
                                            ds_train,
                                            num_replicas=None,
                                            rank=None,
                                            shuffle=shuffle_train)
    # test_sampler = MyNewDistributedSampler(
    #     experiment_manual_seed, ds_test, num_replicas=None, rank=None, shuffle=False)
    test_sampler = None  # FIXME:

    # Note: explicitly set shuffle to False, its handled by samplers.
    dl_train = DataLoader(ds_train,
                          bs_train,
                          shuffle=False,
                          pin_memory=pin_memory,
                          sampler=train_sampler,
                          **kw)
    dl_test = DataLoader(ds_test,
                         bs_test,
                         shuffle=False,
                         pin_memory=pin_memory,
                         sampler=test_sampler,
                         **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test, train_sampler


#############################################
# Distributed. (v3)
# get x separate from y, both with same seed
#############################################


def get_separate_just_x_or_y_test_dl(dataset,
                                     bs_test,
                                     just,
                                     verbose=True,
                                     DATA_DIR=DEFAULT_DATA_DIR,
                                     pin_memory=True,
                                     test_dataset_keywords=dict(),
                                     **kw):
    """ Just the test """
    raise NotImplementedError()

    experiment_manual_seed = torch.initial_seed()

    DICT_DATASET_JUST_XY_FUNC = {'wt2': get_wt2_just_x_or_y_test_ds}

    ds_test = DICT_DATASET_JUST_XY_FUNC.get(dataset)(just=just,
                                                     DATA_DIR=DATA_DIR,
                                                     **test_dataset_keywords)

    test_sampler = MyNewDistributedSampler(experiment_manual_seed,
                                           ds_test,
                                           num_replicas=1,
                                           rank=0,
                                           shuffle=False)

    dl_test = DataLoader(ds_test,
                         bs_test,
                         shuffle=False,
                         pin_memory=pin_memory,
                         sampler=test_sampler,
                         **kw)

    if verbose:
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_test, test_sampler


def get_separate_just_x_or_y_train_test_dl(dataset,
                                           bs_train,
                                           bs_test,
                                           just,
                                           shuffle_train=True,
                                           verbose=True,
                                           DATA_DIR=DEFAULT_DATA_DIR,
                                           pin_memory=True,
                                           dataset_keywords=dict(),
                                           dataloader_keywords=dict(),
                                           **kw):

    experiment_manual_seed = torch.initial_seed()
    print(experiment_manual_seed, "experiment_manual_seed")

    DICT_DATASET_JUST_XY_FUNC = {
        'cifar10': get_cifar_10_just_x_or_y_train_test_ds,
        'cifar100': get_cifar_100_just_x_or_y_train_test_ds,
        'imagenet': get_imagenet_just_x_or_y_train_test_ds,
        'wt2': get_wt2_just_x_or_y_train_test_ds,
        'squad1': get_just_x_or_y_train_dev_dataset_squad,
        'squad2': get_just_x_or_y_train_dev_dataset_squad,
        'glue': get_just_x_or_y_train_dev_dataset_glue,
        't5_squad': get_just_x_or_y_train_dev_dataset_t5_squad,
        # 'wt2': get_wt2_just_x_or_y_train_valid_ds
    }

    ds_train, ds_test, *extra = DICT_DATASET_JUST_XY_FUNC.get(dataset)(
        just=just, DATA_DIR=DATA_DIR, **dataset_keywords)

    # Note: choosing None will infer these args from torch.distributed calls.
    # HACK: we set everything to rank 0 and 1 replica.
    # (we do this to utilize the tested generator code inside the distributed sampler)
    train_sampler = MyNewDistributedSampler(experiment_manual_seed,
                                            ds_train,
                                            num_replicas=1,
                                            rank=0,
                                            shuffle=shuffle_train)

    test_sampler = MyNewDistributedSampler(
        experiment_manual_seed, ds_test, num_replicas=1, rank=0,
        shuffle=False) if ds_test is not None else None

    # Note: explicitly set shuffle to False, its handled by samplers.
    assert 'pin_memory' in dataloader_keywords, str(dataloader_keywords)
    assert 'shuffle' not in dataloader_keywords
    dl_train = DataLoader(
        ds_train,
        bs_train,
        shuffle=False,
        #   pin_memory=pin_memory,
        sampler=train_sampler,
        **dataloader_keywords)
    dl_test = DataLoader(
        ds_test,
        bs_test,
        shuffle=False,
        #  pin_memory=pin_memory,
        sampler=test_sampler,
        **dataloader_keywords) if ds_test is not None else None

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(
            f'Test: {len(dl_test) * bs_test  if dl_test is not None else 0} samples'
        )

    if extra:
        assert len(extra) == 1
        extra = extra[0]

    return dl_train, dl_test, list(filter(
        None, [train_sampler, test_sampler])), extra
