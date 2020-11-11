import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, DatasetFolder
from torchvision.datasets.folder import default_loader

from .datasets import CommonDatasetHandler, register_dataset, register_hardcoded_just_xy_dataset
from .hardcoded_dirs import DEFAULT_DATA_DIR, IMAGENET_ROOT_DIR

DOWNLOAD = False


# NOTE: CIFAR best practice:
# https://github.com/facebookresearch/pycls/tree/master/configs/cifar
# torch.backends.cudnn.benchmark = False

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


##################################################
# Get DS - full dataset (unused, but here anyway)
#################################################

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


########################################
# Get DL (for full dataset: deprecated)
#######################################


def get_cv_train_test_dl(ds_train,
                         ds_test,
                         bs_train,
                         bs_test,
                         shuffle=True,
                         pin_memory=True,
                         **kw):
    # TODO: X to first device and y to last device.
    dl_train = torch.utils.data.DataLoader(ds_train,
                                           bs_train,
                                           shuffle=shuffle,
                                           pin_memory=pin_memory,
                                           **kw)
    dl_test = torch.utils.data.DataLoader(ds_test,
                                          bs_test,
                                          shuffle=False,
                                          pin_memory=pin_memory,
                                          **kw)
    return dl_train, dl_test


#############################################
# get x separate from y, both with same seed
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
        path, _ = self.samples[index]
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


#############################################################
# CIFAR 100
############################################################

def get_cifar_100_just_x_or_y_ds(transform, train, **kw):
    just = kw['just']
    DATA_DIR = kw.get("DATA_DIR", DEFAULT_DATA_DIR)

    just = just.lower()
    if just == 'x':
        ds_X = CIFAR100JustX(root=DATA_DIR,
                             download=DOWNLOAD,
                             train=train,
                             transform=transform)
        return ds_X
    elif just == 'y':
        ds_Y = CIFAR100JustY(root=DATA_DIR,
                             download=DOWNLOAD,
                             train=train,
                             transform=transform)
        return ds_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


class SEP_CIFAR100_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()

    def get_train_ds(self, **kw):
        train_transform, _ = cifar100_transformations()
        return get_cifar_100_just_x_or_y_ds(train_transform, train=True, **kw)

    def get_test_ds(self, *args, **kw):
        _, test_transform = cifar100_transformations()
        return get_cifar_100_just_x_or_y_ds(test_transform, train=False, **kw)

    def get_validation_ds(self, **kw):
        NotImplementedError()


#############################################################
# CIFAR 10
############################################################


def get_cifar_10_just_x_or_y_ds(transform, train, **kw):
    just = kw['just']
    DATA_DIR = kw.get("DATA_DIR", DEFAULT_DATA_DIR)

    if not isinstance(just, str):
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")
    just = just.lower()
    if just == 'x':
        ds_X = CIFAR10JustX(root=DATA_DIR,
                            download=DOWNLOAD,
                            train=train,
                            transform=transform)
        return ds_X
    elif just == 'y':
        ds_Y = CIFAR10JustY(root=DATA_DIR,
                            download=DOWNLOAD,
                            train=train,
                            transform=transform)
        return ds_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


class SEP_CIFAR10_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()

    def get_train_ds(self, **kw):
        train_transform, _ = cifar10_transformations()
        return get_cifar_10_just_x_or_y_ds(transform=train_transform, train=True, **kw)

    def get_test_ds(self, **kw):
        _, test_transform = cifar10_transformations()
        return get_cifar_10_just_x_or_y_ds(transform=test_transform, train=False, **kw)

    def get_validation_ds(self, **kw):
        NotImplementedError()




#############################################################
# IMAGENET
############################################################

def get_imagenet_just_x_or_y_ds(transform, train, **kw):
    just = kw['just']
    DATA_DIR = kw.get("DATA_DIR", DEFAULT_DATA_DIR)

    just = just.lower()
    data_dir = os.path.join(DATA_DIR, 'val') if not train else os.path.join(DATA_DIR, 'train')

    if just == 'x':
        ds_X = ImageFolderJustX(data_dir, transform=transform)
        return ds_X
    elif just == 'y':
        ds_Y = ImageFolderJustY(data_dir, transform=transform, target_transform=None)
        return ds_Y
    else:
        raise ValueError(f"'just' should be in x,y. Got {just} instead.")


class SEP_IMAGENET_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()

    def get_train_ds(self, **kw):
        train_transform, _ = imagenet_transformations()
        return get_imagenet_just_x_or_y_ds(transform=train_transform, train=True, **kw)

    def get_test_ds(self, **kw):
        # For convenience it is given as test, its actually validation
        _, test_transform = imagenet_transformations()
        return get_imagenet_just_x_or_y_ds(transform=test_transform, train=False, **kw)

    def get_validation_ds(self, **kw):
        # For convenience it is given as test, its actually validation
        NotImplementedError()



register_dataset("cifar10", SEP_CIFAR10_DatasetHandler)
register_dataset("cifar100", SEP_CIFAR100_DatasetHandler)
register_dataset("imagenet", SEP_IMAGENET_DatasetHandler)
register_hardcoded_just_xy_dataset("cifar10")
register_hardcoded_just_xy_dataset("cifar100")
register_hardcoded_just_xy_dataset("imagenet")
