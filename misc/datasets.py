import os
import numpy as np

import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image

from torch.utils.data import Dataset, DistributedSampler
import torch.distributed as dist

# new_distributed_get_train_test_dl_from_args
# distributed_get_train_test_dl_from_args
# simplified_get_train_test_dl_from_args
# get_seperate_just_x_or_y_train_test_dl_from_args


# Fallback to this dataset dir of no other dir is given as arument to functions.
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
        super().__init__()
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
    return partition


def distributed_get_train_test_ds(dataset, DATA_DIR=DEFAULT_DATA_DIR, **kw):
    get_dataset_fn = DATASET_TO_DS_FN.get(dataset, None)
    if get_dataset_fn:
        train_ds, test_ds = get_dataset_fn(DATA_DIR=DATA_DIR)
        print(len(train_ds), len(test_ds))
        train_ds = get_dist_partition(train_ds, **kw)
        # NOTE: we do not do dist on test...
        # test_ds = get_dist_partition(test_ds, **kw)
        print(len(train_ds), len(test_ds))
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


def distributed_get_train_test_dl_from_args(args, shuffle_train=True,
                                            verbose=True, num_splits=None, use_split=None, **kw):
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
        g.manual_seed((self.epoch * self.experiment_manual_seed) %
                      self.MAX_INT)
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


def new_distributed_simplified_get_train_test_dl(dataset, bs_train, bs_test, shuffle_train=True, verbose=True,
                                                 DATA_DIR=DEFAULT_DATA_DIR, pin_memory=True, **kw):
    """ Requires:
         that a manual seed is set to the experiment and restorable via torch.initial_seed() """

    ds_train, ds_test = get_train_test_ds(
        dataset, DATA_DIR=DATA_DIR)
    experiment_manual_seed = torch.initial_seed()

    # Note: choosing None will infer these args from torch.distributed calls.
    train_sampler = MyNewDistributedSampler(
        experiment_manual_seed, ds_train, num_replicas=None, rank=None, shuffle=shuffle_train)
    test_sampler = MyNewDistributedSampler(
        experiment_manual_seed, ds_test, num_replicas=None, rank=None, shuffle=False)

    # Note: explicitly set shuffle to False, its handled by samplers.
    dl_train = torch.utils.data.DataLoader(
        ds_train, bs_train, shuffle=False, pin_memory=pin_memory, sampler=train_sampler, **kw)
    dl_test = torch.utils.data.DataLoader(
        ds_test, bs_test, shuffle=False, pin_memory=pin_memory, sampler=test_sampler, **kw)

    # dl_train, dl_test = get_train_test_dl(
    #     dataset, ds_train, ds_test, bs_train, bs_test, shuffle_train=shuffle_train, **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test


def new_distributed_get_train_test_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # num_replicas=None, rank=None
    return new_distributed_simplified_get_train_test_dl(args.dataset, args.bs_train,
                                                        args.bs_test,
                                                        DATA_DIR=DATA_DIR, **kw)


#############################################
# get x seperate from y, both with same seed
#############################################


class CIFAR10JustX(CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: image
        """
        img = self.data[index]
        # target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img  # , target


class CIFAR10JustY(CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: target where target is index of the target class.
        """
        # img = self.data[index]
        target = self.targets[index]

        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target


class CIFAR100JustX(CIFAR10JustX):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class CIFAR100JustY(CIFAR10JustY):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


def get_cifar_100_seperate_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    train_transform, test_transform = cifar_transformations(mean, std)

    ds_train_X = CIFAR100JustX(
        root=DATA_DIR, download=True, train=True, transform=train_transform)
    ds_train_Y = CIFAR100JustY(
        root=DATA_DIR, download=True, train=True, transform=train_transform)

    ds_test_X = CIFAR100JustX(
        root=DATA_DIR, download=True, train=False, transform=test_transform)
    ds_test_Y = CIFAR100JustY(
        root=DATA_DIR, download=True, train=False, transform=test_transform)

    return ds_train_X, ds_train_Y, ds_test_X, ds_test_Y


def get_cifar_10_seperate_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):
    mean = np.array([0.49139968, 0.48215841, 0.44653091])
    std = np.array([0.24703223, 0.24348513, 0.26158784])

    train_transform, test_transform = cifar_transformations(mean, std)

    ds_train_X = CIFAR10JustX(
        root=DATA_DIR, download=True, train=True, transform=train_transform)
    ds_train_Y = CIFAR10JustY(
        root=DATA_DIR, download=True, train=True, transform=train_transform)

    ds_test_X = CIFAR10JustX(root=DATA_DIR, download=True,
                             train=False, transform=test_transform)
    ds_test_Y = CIFAR10JustY(root=DATA_DIR, download=True,
                             train=False, transform=test_transform)

    return ds_train_X, ds_train_Y, ds_test_X, ds_test_Y


def get_cifar_100_just_x_or_y_train_test_ds(just, DATA_DIR=DEFAULT_DATA_DIR):
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    train_transform, test_transform = cifar_transformations(mean, std)
    just = just.lower()
    if just == 'x':
        ds_train_X = CIFAR100JustX(
            root=DATA_DIR, download=True, train=True, transform=train_transform)
        ds_test_X = CIFAR100JustX(
            root=DATA_DIR, download=True, train=False, transform=test_transform)
        return ds_train_X, ds_test_X
    elif just == 'y':
        ds_train_Y = CIFAR100JustY(
            root=DATA_DIR, download=True, train=True, transform=train_transform)
        ds_test_Y = CIFAR100JustY(
            root=DATA_DIR, download=True, train=False, transform=test_transform)
        return ds_train_Y, ds_test_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


def get_cifar_10_just_x_or_y_train_test_ds(just, DATA_DIR=DEFAULT_DATA_DIR):
    mean = np.array([0.49139968, 0.48215841, 0.44653091])
    std = np.array([0.24703223, 0.24348513, 0.26158784])

    train_transform, test_transform = cifar_transformations(mean, std)
    just = just.lower()
    if just == 'x':
        ds_train_X = CIFAR10JustX(
            root=DATA_DIR, download=True, train=True, transform=train_transform)
        ds_test_X = CIFAR10JustX(
            root=DATA_DIR, download=True, train=False, transform=test_transform)
        return ds_train_X, ds_test_X
    elif just == 'y':
        ds_train_Y = CIFAR10JustY(
            root=DATA_DIR, download=True, train=True, transform=train_transform)
        ds_test_Y = CIFAR10JustY(
            root=DATA_DIR, download=True, train=False, transform=test_transform)
        return ds_train_Y, ds_test_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


def get_seperate_just_x_or_y_train_test_dl(dataset, bs_train, bs_test, just,
                                           shuffle_train=True, verbose=True,
                                           DATA_DIR=DEFAULT_DATA_DIR, pin_memory=True, **kw):
    ds_train, ds_test = get_train_test_ds(dataset, DATA_DIR=DATA_DIR)

    DICT_DATASET_JUST_XY_FUNC = {
        'cifar10': get_cifar_10_just_x_or_y_train_test_ds,
        'cifar100': get_cifar_100_just_x_or_y_train_test_ds
    }

    experiment_manual_seed = torch.initial_seed()

    ds_train, ds_test = DICT_DATASET_JUST_XY_FUNC.get(
        dataset)(just=just, DATA_DIR=DATA_DIR)

    # Note: choosing None will infer these args from torch.distributed calls.
    # HACK: we set everything to rank 0 and 1 replica.
    # (we do this to utilize the tested generator code inside the distributed sampler)
    train_sampler = MyNewDistributedSampler(
        experiment_manual_seed, ds_train, num_replicas=1, rank=0, shuffle=shuffle_train)
    test_sampler = MyNewDistributedSampler(
        experiment_manual_seed, ds_test, num_replicas=1, rank=0, shuffle=False)

    # Note: explicitly set shuffle to False, its handled by samplers.
    dl_train = torch.utils.data.DataLoader(
        ds_train, bs_train, shuffle=False, pin_memory=pin_memory, sampler=train_sampler, **kw)
    dl_test = torch.utils.data.DataLoader(
        ds_test, bs_test, shuffle=False, pin_memory=pin_memory, sampler=test_sampler, **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test


def get_seperate_just_x_or_y_train_test_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # Just:
    # HACK: avoid asking "is last partition?"
    just = 'x' if args.stage == 0 else 'y'

    # num_replicas=None, rank=None
    return get_seperate_just_x_or_y_train_test_dl(args.dataset, args.bs_train,
                                                  args.bs_test, just,
                                                  DATA_DIR=DATA_DIR, **kw)
