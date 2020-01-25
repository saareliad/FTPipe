import os
import numpy as np

import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
import torch.distributed as dist


# TODO: remove hardcoded DATA_DIR
# read DATA_DIR from env or some config.yml file.
DEFAULT_DATA_DIR = os.path.expanduser('~/.pytorch-datasets')

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# CIFAR best practice:
# https://github.com/facebookresearch/pycls/tree/master/configs/cifar
# torch.backends.cudnn.benchmark = False


def get_cifar_10_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):
    mean = np.array([0.49139968, 0.48215841, 0.44653091])
    std = np.array([0.24703223, 0.24348513, 0.26158784])

    train_transform, test_transform = cifar_transformations(mean, std)

    ds_train = CIFAR10(root=DATA_DIR, download=True,
                       train=True, transform=train_transform)
    ds_test = CIFAR10(root=DATA_DIR, download=True,
                      train=False, transform=test_transform)
    return ds_train, ds_test


def get_cifar_train_test_dl(ds_train, ds_test, bs_train, bs_test, shuffle_train=True, pin_memory=True, **kw):
    # TODO: X to first device and y to last device.
    dl_train = torch.utils.data.DataLoader(
        ds_train, bs_train, shuffle=shuffle_train, pin_memory=pin_memory, **kw)
    dl_test = torch.utils.data.DataLoader(
        ds_test, bs_test, shuffle=False, pin_memory=pin_memory, **kw)
    return dl_train, dl_test


def get_cifar_100_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    train_transform, test_transform = cifar_transformations(mean, std)

    ds_train = CIFAR100(root=DATA_DIR, download=True,
                        train=True, transform=train_transform)
    ds_test = CIFAR100(root=DATA_DIR, download=True,
                       train=False, transform=test_transform)
    return ds_train, ds_test


def cifar_transformations(mean, std):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    return train_transform, test_transform


DATASET_TO_DS_FN = {
    'cifar10': get_cifar_10_train_test_ds,
    'cifar100': get_cifar_100_train_test_ds
}


DATASET_TO_DL_FN = {
    'cifar10': get_cifar_train_test_dl,
    'cifar100': get_cifar_train_test_dl
}


def get_train_test_ds(dataset, DATA_DIR=DEFAULT_DATA_DIR):
    get_dataset_fn = DATASET_TO_DS_FN.get(dataset, None)
    if get_dataset_fn:
        return get_dataset_fn(DATA_DIR=DATA_DIR)
    else:
        raise ValueError(dataset)


def get_train_test_dl(dataset, *args, **kw):
    get_dl_fn = DATASET_TO_DL_FN.get(dataset, None)
    if get_dl_fn:
        return get_dl_fn(*args, **kw)
    else:
        raise ValueError(dataset)


def simplified_get_train_test_dl(dataset, bs_train, bs_test, shuffle_train=True, verbose=True,
                                 DATA_DIR=DEFAULT_DATA_DIR, **kw):
    ds_train, ds_test = get_train_test_ds(dataset, DATA_DIR=DATA_DIR)

    dl_train, dl_test = get_train_test_dl(
        dataset, ds_train, ds_test, bs_train, bs_test, shuffle_train=shuffle_train, **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test


def simplified_get_train_test_dl_from_args(args, shuffle_train=True, verbose=True, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    return simplified_get_train_test_dl(args.dataset, args.bs_train,
                                        args.bs_test, shuffle_train=shuffle_train, verbose=verbose,
                                        DATA_DIR=DATA_DIR, **kw)


def add_dataset_argument(parser, default='cifar10', required=False):
    parser.add_argument('--dataset', default=default,
                        choices=DATASET_TO_DS_FN.keys(), required=required)


##################
# Distributed...
##################

class Partition(Dataset):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes, seed_to_assert=None):
        # Assert numpy is set with the given seed.
        if not (seed_to_assert is None):
            assert(seed_to_assert == np.random.get_state()[1][0])

        self.data = data
        self.partitions = []

        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        np.random.shuffle(indexes)
        # rng = Random()
        # rng.seed(seed)
        # rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def get_dist_partition(dataset, seed_to_assert=None, num_splits=None, use_split=None):
    if num_splits is None:
        num_splits = dist.get_world_size()
    if use_split is None:
        use_split = dist.get_rank()

    partition_sizes = [1.0 / num_splits for _ in range(num_splits)]
    partition = DataPartitioner(dataset, partition_sizes, seed_to_assert)
    partition = partition.use(use_split)


def distributed_get_train_test_ds(dataset, DATA_DIR=DEFAULT_DATA_DIR, **kw):
    get_dataset_fn = DATASET_TO_DS_FN.get(dataset, None)
    if get_dataset_fn:
        train_ds, test_ds = get_dataset_fn(DATA_DIR=DATA_DIR)
        train_ds = get_dist_partition(train_ds, **kw)
        test_ds = get_dist_partition(test_ds, **kw)
        return train_ds, test_ds
    else:
        raise ValueError(dataset)


def distributed_simplified_get_train_test_dl(dataset, bs_train, bs_test, shuffle_train=True, verbose=True,
                                             DATA_DIR=DEFAULT_DATA_DIR, dist_kw={}, **kw):
    """dist_kw: seed_to_assert=None, num_splits=None, use_split=None """

    ds_train, ds_test = distributed_get_train_test_ds(
        dataset, DATA_DIR=DATA_DIR)

    dl_train, dl_test = get_train_test_dl(
        dataset, ds_train, ds_test, bs_train, bs_test, shuffle_train=shuffle_train, **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test


def distributed_get_train_test_dl_from_args(args, shuffle_train=True, verbose=True, num_splits=None, use_split=None, **kw):
    """
    num_splits: number of splits, that is, num dataparallel gpus (default value is to take dist.world_size).
    use_split: the number of split to use on this proccess (default value is to take dist.rank)
    """
    # TODO: num_splits=None, use_split=None
    dist_kw = dict(seed_to_assert=args.seed,
                   num_splits=num_splits, use_split=use_split)

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    return distributed_simplified_get_train_test_dl(args.dataset, args.bs_train,
                                                    args.bs_test, shuffle_train=shuffle_train, verbose=verbose,
                                                    DATA_DIR=DATA_DIR, dist_kw=dist_kw, **kw)
