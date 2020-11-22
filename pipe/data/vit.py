"""This file will handle data preperation\preproc fror ViT (Vision Transformer) experiments
# See https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py#L110
"""
import numpy as np
import torch
import torchvision

from pipe.data import CommonDatasetHandler, register_dataset, register_hardcoded_just_xy_dataset
from pipe.data.cv import get_imagenet_just_x_or_y_ds, get_cifar_10_just_x_or_y_ds, get_cifar_100_just_x_or_y_ds

#
# IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
# CIFAR100_DEFAULT_MEAN = [0.5071, 0.4867, 0.4408]
# CIFAR100_DEFAULT_STD = [0.2675, 0.2565, 0.2761]
# CIFAR10_DEFAULT_MEAN = [0.49139968, 0.48215841, 0.44653091]
# CIFAR10_DEFAULT_STD = [0.24703223, 0.24348513, 0.26158784]
#

IMAGENET_DEFAULT_MEAN = [0.5, 0.5, 0.5]
IMAGENET_DEFAULT_STD = [0.5, 0.5, 0.5]
CIFAR100_DEFAULT_MEAN = [0.5, 0.5, 0.5]
CIFAR100_DEFAULT_STD = [0.5, 0.5, 0.5]
CIFAR10_DEFAULT_MEAN = [0.5, 0.5, 0.5]
CIFAR10_DEFAULT_STD = [0.5, 0.5, 0.5]


def get_transformations(mean, std, resize_size, crop_size, mode='train', jit_script=False):
    if mode == 'train':
        transform = [
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.RandomCrop((crop_size, crop_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # Note: replacing:     im = (im - 127.5) / 127.5  of original
            torchvision.transforms.Normalize(mean, std),
        ]
    else:
        transform = [
            # usage of crop_size here is intentional
            torchvision.transforms.Resize((crop_size, crop_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]

    if jit_script:
        transform = torch.nn.Sequential(*transform)
        transform = torch.jit.script(transform)
    else:
        transform = torchvision.transforms.Compose(transform)

    return transform


def cifar10_transformations(jit_script=False, resize_size=384, crop_size=384):
    mean = np.array(CIFAR10_DEFAULT_MEAN)
    std = np.array(CIFAR10_DEFAULT_STD)
    train_transform = get_transformations(mean=mean, std=std, crop_size=crop_size, resize_size=resize_size,
                                          mode='train', jit_script=jit_script)
    test_transform = get_transformations(mean=mean, std=std, crop_size=crop_size, resize_size=resize_size, mode='test',
                                         jit_script=jit_script)
    return train_transform, test_transform


def cifar100_transformations(jit_script=False, resize_size=384, crop_size=384):
    mean = np.array(CIFAR100_DEFAULT_MEAN)
    std = np.array(CIFAR100_DEFAULT_STD)
    train_transform = get_transformations(mean=mean, std=std, crop_size=crop_size, resize_size=resize_size,
                                          mode='train', jit_script=jit_script)
    test_transform = get_transformations(mean=mean, std=std, crop_size=crop_size, resize_size=resize_size, mode='test',
                                         jit_script=jit_script)
    return train_transform, test_transform


def imagenet_transformations(jit_script=False, resize_size=384, crop_size=384):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    train_transform = get_transformations(mean=mean, std=std, crop_size=crop_size, resize_size=resize_size,
                                          mode='train', jit_script=jit_script)
    test_transform = get_transformations(mean=mean, std=std, crop_size=crop_size, resize_size=resize_size, mode='test',
                                         jit_script=jit_script)
    return train_transform, test_transform


def sep_imagenet_handler_factory(resize_size=384, crop_size=384):
    class SepImagenetAutoGenDatasetHandler(CommonDatasetHandler):
        def __init__(self, **kw):
            super().__init__()

        def get_train_ds(self, **kw):
            train_transform, _ = imagenet_transformations(resize_size=resize_size, crop_size=crop_size)
            return get_imagenet_just_x_or_y_ds(transform=train_transform, train=True, **kw)

        def get_test_ds(self, **kw):
            # For convenience it is given as test, its actually validation
            _, test_transform = imagenet_transformations(resize_size=resize_size, crop_size=crop_size)
            return get_imagenet_just_x_or_y_ds(transform=test_transform, train=False, **kw)

        def get_validation_ds(self, **kw):
            # For convenience it is given as test, its actually validation
            NotImplementedError()

    return SepImagenetAutoGenDatasetHandler


class SepCifar10_384_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()

    def get_train_ds(self, **kw):
        train_transform, _ = cifar10_transformations(resize_size=384, crop_size=384)
        return get_cifar_10_just_x_or_y_ds(transform=train_transform, train=True, **kw)

    def get_test_ds(self, **kw):
        _, test_transform = cifar10_transformations(resize_size=384, crop_size=384)
        return get_cifar_10_just_x_or_y_ds(transform=test_transform, train=False, **kw)

    def get_validation_ds(self, **kw):
        NotImplementedError()


class SepCifar100_384_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()

    def get_train_ds(self, **kw):
        train_transform, _ = cifar100_transformations(resize_size=384, crop_size=384)
        return get_cifar_100_just_x_or_y_ds(transform=train_transform, train=True, **kw)

    def get_test_ds(self, **kw):
        _, test_transform = cifar100_transformations(resize_size=384, crop_size=384)
        return get_cifar_100_just_x_or_y_ds(transform=test_transform, train=False, **kw)

    def get_validation_ds(self, **kw):
        NotImplementedError()


# For ImageNet results in Table 2, we fine-tuned at higher resolution:
# 512 for ViT-L/16 and 518 for ViT-H/14

# TODO: Polyak & Juditsky (1992) averaging with a factor of 0.9999 (Ramachandran et al., 2019; Wang et al., 2020b).
# TODO: For ImageNet we found it beneficial to additionally apply gradient clipping at global norm 1

register_dataset("cifar10_384", SepCifar10_384_DatasetHandler)
register_dataset("cifar100_384", SepCifar100_384_DatasetHandler)
register_dataset("imagenet_384", sep_imagenet_handler_factory(resize_size=384, crop_size=384))
register_dataset("imagenet_512", sep_imagenet_handler_factory(resize_size=512, crop_size=512))
register_dataset("imagenet_518", sep_imagenet_handler_factory(resize_size=518, crop_size=518))

register_hardcoded_just_xy_dataset("cifar10_384")
register_hardcoded_just_xy_dataset("cifar100_384")
register_hardcoded_just_xy_dataset("imagenet_384")
register_hardcoded_just_xy_dataset("imagenet_512")
register_hardcoded_just_xy_dataset("imagenet_518")
