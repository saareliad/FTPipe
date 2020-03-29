import os

import torch

from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
# import torch.distributed as dist

# new_distributed_get_train_valid_dl_from_args  (train, valid)
# simplified_get_train_valid_dl_from_args  (train, valid)
# get_separate_just_x_or_y_train_test_dl_from_args  (train, valid)
# get_separate_just_x_or_y_test_dl_from_args: (just the test dataloader)

# Fallback to this dataset dir of no other dir is given as argument to functions.
DEFAULT_DATA_DIR = os.path.expanduser('~/.pytorch-datasets')
IMAGENET_ROOT_DIR = "/home_local/saareliad/data/imagenet/"
DOWNLOAD = False
# WIKI2_DATA_DIR = DATA_DIR/wikitext-2-raw
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# CIFAR best practice:
# https://github.com/facebookresearch/pycls/tree/master/configs/cifar
# torch.backends.cudnn.benchmark = False

AVAILABLE_DATASETS = {'cifar10', 'cifar100', 'imagenet', 'wt2'}

############################
# Forward decalted Datasets # FIXME
############################

from .lm import TextDataset

################
# Transforms
################
from .cv import cifar_transformations, cifar10_transformations, cifar100_transformations, imagenet_transformations
from .lm import mask_tokens, mask_tokens_just_inputs, mask_tokens_just_labels

################
# Get DS
################

from .cv import get_cifar_100_train_test_ds, get_imagenet_train_test_ds, get_cifar_10_train_test_ds

from .lm import (get_wikitext2_raw_train_valid_test_ds,
                 get_wikitext2_raw_train_test_ds,
                 get_wikitext2_raw_train_valid_ds, get_wikitext2_raw_test_ds)

# NOTE: these are functions which returns train and test/validation datasets.
DATASET_TO_DS_FN = {
    'cifar10': get_cifar_10_train_test_ds,
    'cifar100': get_cifar_100_train_test_ds,
    'imagenet': get_imagenet_train_test_ds,
    'wt2': get_wikitext2_raw_train_test_ds,  # TODO
    # 'wt2': get_wikitext2_raw_train_valid_ds,  # TODO
}

################
# Get DL
################

from .cv import get_cv_train_test_dl

from .lm import lm_collate_factory, get_lm_train_dl, get_lm_eval_dl, get_lm_train_valid_dl

# NOTE: functions which returns 2 dataloaders, train and valid/test
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


# TODO : this works just for CV
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


###################################
# Dataset from args and key words.
###################################


def add_dataset_argument(parser, default='cifar10', required=False):
    parser.add_argument('--dataset',
                        default=default,
                        choices=list(AVAILABLE_DATASETS),
                        required=required)


def args_extractor1(args):
    """extracts:
        args.dataset, args.bs_train, args.bs_test, args.data_dir
    """
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR
    return dict(DATA_DIR=DATA_DIR,
                dataset=args.dataset,
                bs_train=args.bs_train,
                bs_test=args.bs_test)


def simplified_get_train_valid_dl_from_args(args,
                                            shuffle_train=True,
                                            verbose=True,
                                            **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    return simplified_get_train_test_dl(args.dataset,
                                        args.bs_train,
                                        args.bs_test,
                                        shuffle_train=shuffle_train,
                                        verbose=verbose,
                                        DATA_DIR=DATA_DIR,
                                        **kw)


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
            (self.epoch * self.experiment_manual_seed) % self.MAX_INT)
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
    dl_train = torch.utils.data.DataLoader(ds_train,
                                           bs_train,
                                           shuffle=False,
                                           pin_memory=pin_memory,
                                           sampler=train_sampler,
                                           **kw)
    dl_test = torch.utils.data.DataLoader(ds_test,
                                          bs_test,
                                          shuffle=False,
                                          pin_memory=pin_memory,
                                          sampler=test_sampler,
                                          **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test, train_sampler


def new_distributed_get_train_valid_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # HACK create collate if needed... FIXME TODO
    # TODO: move this to the use code.
    if 'dataset_keywords' in kw:
        dataset_keywords = kw['dataset_keywords']
        if 'tokenizer' in dataset_keywords:
            tokenizer = dataset_keywords['tokenizer']
            collate = lm_collate_factory(tokenizer)
            kw['collate'] = collate

    # num_replicas=None, rank=None
    return new_distributed_simplified_get_train_test_dl(args.dataset,
                                                        args.bs_train,
                                                        args.bs_test,
                                                        DATA_DIR=DATA_DIR,
                                                        **kw)


#############################################
# get x separate from y, both with same seed
#############################################
from .cv import (get_cifar_10_just_x_or_y_train_test_ds,
                 get_imagenet_just_x_or_y_train_test_ds,
                 get_cifar_100_just_x_or_y_train_test_ds,
                 get_cifar_10_separate_train_test_ds,
                 get_cifar_100_separate_train_test_ds, CIFAR100JustY,
                 CIFAR10JustY, CIFAR10JustX, CIFAR100JustX, ImageFolderJustY,
                 ImageFolderJustX, DatasetFolderJustY, DatasetFolderJustX)
from .lm import (get_wt2_just_x_or_y_train_valid_ds,
                 get_wt2_just_x_or_y_train_test_ds,
                 get_wt2_just_x_or_y_test_ds)


def get_separate_just_x_or_y_test_dl(dataset,
                                     bs_test,
                                     just,
                                     verbose=True,
                                     DATA_DIR=DEFAULT_DATA_DIR,
                                     pin_memory=True,
                                     test_dataset_keywords=dict(),
                                     **kw):
    """ Just the test """

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

    # get_lm_eval_dl

    dl_test = torch.utils.data.DataLoader(ds_test,
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
                                           **kw):

    experiment_manual_seed = torch.initial_seed()

    DICT_DATASET_JUST_XY_FUNC = {
        'cifar10': get_cifar_10_just_x_or_y_train_test_ds,
        'cifar100': get_cifar_100_just_x_or_y_train_test_ds,
        'imagenet': get_imagenet_just_x_or_y_train_test_ds,
        'wt2': get_wt2_just_x_or_y_train_test_ds
        # 'wt2': get_wt2_just_x_or_y_train_valid_ds
    }

    ds_train, ds_test = DICT_DATASET_JUST_XY_FUNC.get(dataset)(
        just=just, DATA_DIR=DATA_DIR, **dataset_keywords)

    # Note: choosing None will infer these args from torch.distributed calls.
    # HACK: we set everything to rank 0 and 1 replica.
    # (we do this to utilize the tested generator code inside the distributed sampler)
    train_sampler = MyNewDistributedSampler(experiment_manual_seed,
                                            ds_train,
                                            num_replicas=1,
                                            rank=0,
                                            shuffle=shuffle_train)
    test_sampler = MyNewDistributedSampler(experiment_manual_seed,
                                           ds_test,
                                           num_replicas=1,
                                           rank=0,
                                           shuffle=False)

    # Note: explicitly set shuffle to False, its handled by samplers.
    dl_train = torch.utils.data.DataLoader(ds_train,
                                           bs_train,
                                           shuffle=False,
                                           pin_memory=pin_memory,
                                           sampler=train_sampler,
                                           **kw)
    dl_test = torch.utils.data.DataLoader(ds_test,
                                          bs_test,
                                          shuffle=False,
                                          pin_memory=pin_memory,
                                          sampler=test_sampler,
                                          **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test, [train_sampler, test_sampler]


################
# From Args
###############

def get_separate_just_x_or_y_train_test_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # Just:
    # HACK: avoid asking "is last partition?"
    just = 'x' if args.stage == 0 else 'y'

    # num_replicas=None, rank=None
    return get_separate_just_x_or_y_train_test_dl(
        args.dataset,
        args.bs_train,
        # TODO: change it to validation...
        args.bs_test,
        just,
        DATA_DIR=DATA_DIR,
        **kw)


def get_separate_just_x_or_y_test_dl_from_args(args, **kw):
    """ get just the test dataset.
    kw can have
    test_dataset_keywords=dict()
    to help with it
    """
    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # Just:
    # HACK: avoid asking "is last partition?"
    just = 'x' if args.stage == 0 else 'y'

    # num_replicas=None, rank=None
    return get_separate_just_x_or_y_test_dl(args.dataset,
                                            args.bs_test,
                                            just,
                                            DATA_DIR=DATA_DIR,
                                            **kw)
