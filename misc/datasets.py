import os
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, DatasetFolder
from torchvision.datasets.folder import default_loader
from PIL import Image

from torch.utils.data import Dataset, DistributedSampler
import torch.distributed as dist

# new_distributed_get_train_test_dl_from_args
# distributed_get_train_test_dl_from_args
# simplified_get_train_test_dl_from_args
# get_seperate_just_x_or_y_train_test_dl_from_args

# Fallback to this dataset dir of no other dir is given as arument to functions.
DEFAULT_DATA_DIR = os.path.expanduser('~/.pytorch-datasets')
IMAGENET_ROOT_DIR = "/home_local/saareliad/data/imagenet/"
DOWNLOAD = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# CIFAR best practice:
# https://github.com/facebookresearch/pycls/tree/master/configs/cifar
# torch.backends.cudnn.benchmark = False

AVAILABLE_DATASETS = {'cifar10', 'cifar100', 'imagenet'}

################
# Transforms
################


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


def cifar10_transformations():
    mean = np.array([0.49139968, 0.48215841, 0.44653091])
    std = np.array([0.24703223, 0.24348513, 0.26158784])
    train_transform, test_transform = cifar_transformations(mean, std)
    return train_transform, test_transform


def cifar100_transformations():
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    train_transform, test_transform = cifar_transformations(mean, std)
    return train_transform, test_transform


def imagenet_transformations():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


################
# Get DS
################


def get_cifar_10_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):
    train_transform, test_transform = cifar10_transformations()

    ds_train = CIFAR10(root=DATA_DIR,
                       download=DOWNLOAD,
                       train=True,
                       transform=train_transform)
    ds_test = CIFAR10(root=DATA_DIR,
                      download=DOWNLOAD,
                      train=False,
                      transform=test_transform)
    return ds_train, ds_test


def get_imagenet_train_test_ds(DATA_DIR=IMAGENET_ROOT_DIR):
    train_transform, test_transform = imagenet_transformations()
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    ds_train = ImageFolder(traindir, transform=train_transform)
    ds_test = ImageFolder(valdir, transform=test_transform)

    return ds_train, ds_test


def get_cifar_100_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):
    train_transform, test_transform = cifar100_transformations()

    ds_train = CIFAR100(root=DATA_DIR,
                        download=DOWNLOAD,
                        train=True,
                        transform=train_transform)
    ds_test = CIFAR100(root=DATA_DIR,
                       download=DOWNLOAD,
                       train=False,
                       transform=test_transform)
    return ds_train, ds_test


DATASET_TO_DS_FN = {
    'cifar10': get_cifar_10_train_test_ds,
    'cifar100': get_cifar_100_train_test_ds,
    'imagenet': get_imagenet_train_test_ds,
}

################
# Get DL
################


def get_cv_train_test_dl(ds_train,
                         ds_test,
                         bs_train,
                         bs_test,
                         shuffle_train=True,
                         pin_memory=True,
                         **kw):
    # TODO: X to first device and y to last device.
    dl_train = torch.utils.data.DataLoader(ds_train,
                                           bs_train,
                                           shuffle=shuffle_train,
                                           pin_memory=pin_memory,
                                           **kw)
    dl_test = torch.utils.data.DataLoader(ds_test,
                                          bs_test,
                                          shuffle=False,
                                          pin_memory=pin_memory,
                                          **kw)
    return dl_train, dl_test


DATASET_TO_DL_FN = {
    'cifar10': get_cv_train_test_dl,
    'cifar100': get_cv_train_test_dl,
    'imagenet': get_cv_train_test_dl
}

# DICT_DATASET_JUST_XY_FUNC = {
#     'cifar10': get_cifar_10_just_x_or_y_train_test_ds,
#     'cifar100': get_cifar_100_just_x_or_y_train_test_ds
# }

############################
# Generic "get" functions
############################


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


############################
# Simpplified. dataset by name.
############################


def simplified_get_train_test_dl(dataset,
                                 bs_train,
                                 bs_test,
                                 shuffle_train=True,
                                 verbose=True,
                                 DATA_DIR=DEFAULT_DATA_DIR,
                                 **kw):
    ds_train, ds_test = get_train_test_ds(dataset, DATA_DIR=DATA_DIR)

    dl_train, dl_test = get_train_test_dl(dataset,
                                          ds_train,
                                          ds_test,
                                          bs_train,
                                          bs_test,
                                          shuffle_train=shuffle_train,
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


def simplified_get_train_test_dl_from_args(args,
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
                                                 **kw):
    """ Requires:
         that a manual seed is set to the experiment and restorable via torch.initial_seed() """

    ds_train, ds_test = get_train_test_ds(dataset, DATA_DIR=DATA_DIR)
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

    # dl_train, dl_test = get_train_test_dl(
    #     dataset, ds_train, ds_test, bs_train, bs_test, shuffle_train=shuffle_train, **kw)

    if verbose:
        print(f'Train: {len(dl_train) * bs_train} samples')
        print(f'Test: {len(dl_test) * bs_test} samples')

    return dl_train, dl_test, train_sampler


def new_distributed_get_train_test_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # num_replicas=None, rank=None
    return new_distributed_simplified_get_train_test_dl(args.dataset,
                                                        args.bs_train,
                                                        args.bs_test,
                                                        DATA_DIR=DATA_DIR,
                                                        **kw)


#############################################
# get x seperate from y, both with same seed
#############################################


class DatasetFolderJustX(DatasetFolder):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class DatasetFolderJustY(DatasetFolder):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            target where target is class_index of the target class.
        """
        _, target = self.samples[index]
        # sample = self.loader(path)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


class ImageFolderJustX(DatasetFolderJustX):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None):
        super().__init__(root,
                         loader,
                         IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.imgs = self.samples


class ImageFolderJustY(DatasetFolderJustY):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 is_valid_file=None):
        super().__init__(root,
                         loader,
                         IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.imgs = self.samples


class CIFAR10JustX(CIFAR10):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

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
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

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
    train_transform, test_transform = cifar100_transformations()

    ds_train_X = CIFAR100JustX(root=DATA_DIR,
                               download=DOWNLOAD,
                               train=True,
                               transform=train_transform)
    ds_train_Y = CIFAR100JustY(root=DATA_DIR,
                               download=DOWNLOAD,
                               train=True,
                               transform=train_transform)

    ds_test_X = CIFAR100JustX(root=DATA_DIR,
                              download=DOWNLOAD,
                              train=False,
                              transform=test_transform)
    ds_test_Y = CIFAR100JustY(root=DATA_DIR,
                              download=DOWNLOAD,
                              train=False,
                              transform=test_transform)

    return ds_train_X, ds_train_Y, ds_test_X, ds_test_Y


def get_cifar_10_seperate_train_test_ds(DATA_DIR=DEFAULT_DATA_DIR):

    train_transform, test_transform = cifar10_transformations()

    ds_train_X = CIFAR10JustX(root=DATA_DIR,
                              download=DOWNLOAD,
                              train=True,
                              transform=train_transform)
    ds_train_Y = CIFAR10JustY(root=DATA_DIR,
                              download=DOWNLOAD,
                              train=True,
                              transform=train_transform)

    ds_test_X = CIFAR10JustX(root=DATA_DIR,
                             download=DOWNLOAD,
                             train=False,
                             transform=test_transform)
    ds_test_Y = CIFAR10JustY(root=DATA_DIR,
                             download=DOWNLOAD,
                             train=False,
                             transform=test_transform)

    return ds_train_X, ds_train_Y, ds_test_X, ds_test_Y


def get_cifar_100_just_x_or_y_train_test_ds(just, DATA_DIR=DEFAULT_DATA_DIR):
    train_transform, test_transform = cifar100_transformations()
    just = just.lower()
    if just == 'x':
        ds_train_X = CIFAR100JustX(root=DATA_DIR,
                                   download=DOWNLOAD,
                                   train=True,
                                   transform=train_transform)
        ds_test_X = CIFAR100JustX(root=DATA_DIR,
                                  download=DOWNLOAD,
                                  train=False,
                                  transform=test_transform)
        return ds_train_X, ds_test_X
    elif just == 'y':
        ds_train_Y = CIFAR100JustY(root=DATA_DIR,
                                   download=DOWNLOAD,
                                   train=True,
                                   transform=train_transform)
        ds_test_Y = CIFAR100JustY(root=DATA_DIR,
                                  download=DOWNLOAD,
                                  train=False,
                                  transform=test_transform)
        return ds_train_Y, ds_test_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


def get_imagenet_just_x_or_y_train_test_ds(just, DATA_DIR=IMAGENET_ROOT_DIR):
    train_transform, test_transform = imagenet_transformations()
    just = just.lower()
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    if just == 'x':
        ds_train_X = ImageFolderJustX(traindir, transform=train_transform)
        ds_test_X = ImageFolderJustX(valdir, transform=test_transform)
        return ds_train_X, ds_test_X
    elif just == 'y':
        ds_train_Y = ImageFolderJustY(traindir, transform=train_transform)
        ds_test_Y = ImageFolderJustY(valdir, transform=test_transform)
        return ds_train_Y, ds_test_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


def get_cifar_10_just_x_or_y_train_test_ds(just, DATA_DIR=DEFAULT_DATA_DIR):
    train_transform, test_transform = cifar10_transformations()
    just = just.lower()
    if just == 'x':
        ds_train_X = CIFAR10JustX(root=DATA_DIR,
                                  download=DOWNLOAD,
                                  train=True,
                                  transform=train_transform)
        ds_test_X = CIFAR10JustX(root=DATA_DIR,
                                 download=DOWNLOAD,
                                 train=False,
                                 transform=test_transform)
        return ds_train_X, ds_test_X
    elif just == 'y':
        ds_train_Y = CIFAR10JustY(root=DATA_DIR,
                                  download=DOWNLOAD,
                                  train=True,
                                  transform=train_transform)
        ds_test_Y = CIFAR10JustY(root=DATA_DIR,
                                 download=DOWNLOAD,
                                 train=False,
                                 transform=test_transform)
        return ds_train_Y, ds_test_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


def get_seperate_just_x_or_y_train_test_dl(dataset,
                                           bs_train,
                                           bs_test,
                                           just,
                                           shuffle_train=True,
                                           verbose=True,
                                           DATA_DIR=DEFAULT_DATA_DIR,
                                           pin_memory=True,
                                           **kw):

    # ds_train, ds_test = get_train_test_ds(dataset, DATA_DIR=DATA_DIR)

    experiment_manual_seed = torch.initial_seed()

    DICT_DATASET_JUST_XY_FUNC = {
        'cifar10': get_cifar_10_just_x_or_y_train_test_ds,
        'cifar100': get_cifar_100_just_x_or_y_train_test_ds,
        'imagenet': get_imagenet_just_x_or_y_train_test_ds
    }

    ds_train, ds_test = DICT_DATASET_JUST_XY_FUNC.get(dataset)(
        just=just, DATA_DIR=DATA_DIR)

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


def get_seperate_just_x_or_y_train_test_dl_from_args(args, **kw):

    DATA_DIR = getattr(args, "data_dir", DEFAULT_DATA_DIR)
    DATA_DIR = DATA_DIR if DATA_DIR else DEFAULT_DATA_DIR

    # Just:
    # HACK: avoid asking "is last partition?"
    just = 'x' if args.stage == 0 else 'y'

    # num_replicas=None, rank=None
    return get_seperate_just_x_or_y_train_test_dl(args.dataset,
                                                  args.bs_train,
                                                  args.bs_test,
                                                  just,
                                                  DATA_DIR=DATA_DIR,
                                                  **kw)
