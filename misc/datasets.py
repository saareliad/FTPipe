import os
import numpy as np

import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100

DATA_DIR = os.path.expanduser('~/.pytorch-datasets')

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# CIFAR best practice:
# https://github.com/facebookresearch/pycls/tree/master/configs/cifar
# torch.backends.cudnn.benchmark = False


def get_cifar_10_train_test_ds():
    mean = np.array([0.49139968, 0.48215841, 0.44653091])
    std = np.array([0.24703223, 0.24348513, 0.26158784])

    train_transform, test_transform = cifar_transformations(mean, std)

    ds_train = CIFAR10(root=DATA_DIR, download=True,
                       train=True, transform=train_transform)
    ds_test = CIFAR10(root=DATA_DIR, download=True,
                      train=False, transform=test_transform)
    return ds_train, ds_test


def get_cifar_train_test_dl(ds_train, ds_test, bs_train, bs_test, shuffle_train=True, pin_memory=True):
    # TODO: X to first device and y to last device.
    dl_train = torch.utils.data.DataLoader(
        ds_train, bs_train, shuffle=shuffle_train, pin_memory=pin_memory)
    dl_test = torch.utils.data.DataLoader(
        ds_test, bs_test, shuffle=False, pin_memory=pin_memory)
    return dl_train, dl_test

def get_cifar_100_train_test_ds():
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


def get_train_test_ds(dataset):
    get_dataset_fn = DATASET_TO_DS_FN.get(dataset, None)
    if get_dataset_fn:
        return get_dataset_fn()
    else:
        raise ValueError(dataset)


def get_train_test_dl(dataset, *args, **kw):
    get_dl_fn = DATASET_TO_DL_FN.get(dataset, None)
    if get_dl_fn:
        return get_dl_fn(*args, **kw)
    else:
        raise ValueError(dataset)

def simplified_get_train_test_dl(dataset, bs_train, bs_test, shuffle_train=True, verbose=True):
    ds_train, ds_test = get_train_test_ds(dataset)

    dl_train, dl_test = get_train_test_dl(
        dataset, ds_train, ds_test, bs_train, bs_test, shuffle_train=shuffle_train)
    
    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test

def simplified_get_train_test_dl_from_args(args, shuffle_train=True, verbose=True):
    return simplified_get_train_test_dl(args.dataset, args.bs_train, args.bs_test, shuffle_train=True, verbose=True)


def add_dataset_argument(parser, default='cifar10', required=False):
    parser.add_argument('--dataset', default=default,
                        choices=DATASET_TO_DS_FN.keys(), required=required)