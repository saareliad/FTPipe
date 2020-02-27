import os
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, DatasetFolder
from torchvision.datasets.folder import default_loader
from PIL import Image

from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
import torch.distributed as dist
import pickle
from transformers import PreTrainedTokenizer

from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

# new_distributed_get_train_test_dl_from_args
# simplified_get_train_test_dl_from_args
# get_seperate_just_x_or_y_train_test_dl_from_args

# Fallback to this dataset dir of no other dir is given as arument to functions.
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


class TextDataset(Dataset):
    # Dataset adapted from huggingface LM example.
    # https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
    # (but without the args thing...)
    def __init__(self,
                 tokenizer,
                 model_name_or_path,
                 overwrite_cache=False,
                 file_path='train',
                 block_size=512):
        assert os.path.isfile(file_path), file_path
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, model_name_or_path + '_cached_lm_' + str(block_size) +
            '_' + filename)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)

        else:
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                # NOTE: this makes it is sutible mostly for small datasets...
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text))

            # Truncate in block of block_size
            for i in range(0,
                           len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i:i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


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


# NOTE: This is like a "transform", Should be used stright in the dataset,
# so the dataloader will handle this.
# NOTE: we also provide 2 more functions just for inputs/labels
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
# commitid: f54a5bd37f99e3933a396836cb0be0b5a497c077
def mask_tokens(inputs: torch.Tensor,
                tokenizer: PreTrainedTokenizer,
                mlm_probability=0.15,
                generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling:
     80% MASK, 10% random, 10% original.

     Usage:
        inputs, labels = mask_tokens(batch, tokenizer, args.mlm_probability, generator) if args.mlm else (batch, batch)
     """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer.")

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                 dtype=torch.bool),
                                    value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix,
                                     generator=generator).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape,
                   0.8), generator=generator).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer),
                                 labels.shape,
                                 dtype=torch.long,
                                 generator=generator)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def mask_tokens_just_inputs(inputs: torch.Tensor,
                            tokenizer: PreTrainedTokenizer,
                            mlm_probability=0.15,
                            generator=None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling:
     80% MASK, 10% random, 10% original.

     Usage:
        inputs, labels = mask_tokens(batch, tokenizer, args.mlm_probability, generator) if args.mlm else (batch, batch)
     """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer.")

    labels = inputs  # HACK: to change less code...
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                 dtype=torch.bool),
                                    value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix,
                                     generator=generator).bool()
    # NOTE: line below removed:
    # labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape,
                   0.8), generator=generator).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer),
                                 labels.shape,
                                 dtype=torch.long,
                                 generator=generator)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs  # NOTE: returning just inputs.


def mask_tokens_just_labels(inputs: torch.Tensor,
                            tokenizer: PreTrainedTokenizer,
                            mlm_probability=0.15,
                            generator=None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling:
     80% MASK, 10% random, 10% original.

     Usage:
        inputs, labels = mask_tokens(batch, tokenizer, args.mlm_probability, generator) if args.mlm else (batch, batch)
     """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer.")

    labels = inputs  # HACK: to change less code.
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                 dtype=torch.bool),
                                    value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix,
                                     generator=generator).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # NOTE: deleted code lines not concering labels.

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return labels  # NOTE: returning just labels.


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


def get_wikitext2_raw_train_valid_test_ds(
        model_name_or_path,
        tokenizer,
        block_size=512,  # NOTE: This is the sequence length
        overwrite_cache=False,
        DATA_DIR=DEFAULT_DATA_DIR,
        split='all'):
    wt2_data_path = os.path.join(DATA_DIR, "wikitext-2-raw")
    train_file = os.path.join(wt2_data_path, "wiki.train.raw")
    valid_file = os.path.join(wt2_data_path, "wiki.valid.raw")
    test_file = os.path.join(wt2_data_path, "wiki.test.raw")

    def get_ds(file_path):
        return TextDataset(tokenizer,
                           model_name_or_path,
                           overwrite_cache=overwrite_cache,
                           file_path=file_path,
                           block_size=block_size)

    if split == 'all':
        train_ds = get_ds(train_file)
        valid_ds = get_ds(valid_file)
        test_ds = get_ds(test_file)
        return train_ds, valid_ds, test_ds
    elif split == 'train':
        train_ds = get_ds(train_file)
        return train_ds
    elif split == 'valid':
        valid_ds = get_ds(valid_file)
        return valid_ds
    elif split == 'test':
        test_ds = get_ds(test_file)
        return test_ds
    else:
        raise ValueError(f"Unsupported split {split}.")


def get_wikitext2_raw_train_valid_ds(model_name_or_path,
                                     tokenizer,
                                     train_seq_len=512,
                                     valid_seq_len=512,
                                     overwrite_cache=False,
                                     DATA_DIR=DEFAULT_DATA_DIR):

    train_ds = get_wikitext2_raw_train_valid_test_ds(
        split='train',
        block_size=train_seq_len,
        overwrite_cache=overwrite_cache)
    valid_ds = get_wikitext2_raw_train_valid_test_ds(
        split='valid',
        block_size=valid_seq_len,
        overwrite_cache=overwrite_cache)
    return train_ds, valid_ds


DATASET_TO_DS_FN = {
    'cifar10': get_cifar_10_train_test_ds,
    'cifar100': get_cifar_100_train_test_ds,
    'imagenet': get_imagenet_train_test_ds,
    'wt2': get_wikitext2_raw_train_valid_ds,  # TODO
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


def lm_collate_factory(tokenizer):
    assert tokenizer is not None

    def lm_collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples,
                            batch_first=True,
                            padding_value=tokenizer.pad_token_id)

    return lm_collate


def get_lm_train_dl(ds_train, bs_train, tokenizer=None, collate_fn=None):
    collate = collate_fn if collate_fn else lm_collate_factory(tokenizer)
    train_sampler = RandomSampler(ds_train)
    train_dl = DataLoader(ds_train,
                          sampler=train_sampler,
                          batch_size=bs_train,
                          collate_fn=collate)
    return train_dl


def get_lm_eval_dl(ds_eval, bs_eval, tokenizer=None, collate_fn=None):
    collate = collate_fn if collate_fn else lm_collate_factory(tokenizer)
    eval_sampler = SequentialSampler(ds_eval)
    eval_dl = DataLoader(bs_eval,
                         sampler=eval_sampler,
                         batch_size=bs_eval,
                         collate_fn=collate)
    return eval_dl


def get_lm_train_valid_dl(ds_train, ds_test, bs_train, bs_test,
                          tokenizer=None):
    # HACK: tokenizer as kwarg.
    # HACK: parameters names are 'test' for backward compatability.
    collate = lm_collate_factory(tokenizer)

    train_dl = get_lm_train_dl(ds_train, bs_train, collate_fn=collate)
    valid_dl = get_lm_eval_dl(ds_test, bs_test, collate_fn=collate)

    return train_dl, valid_dl


#     # UNUSED
# def get_lm_train_valid_test_dl(ds_train, ds_valid, ds_test,
#                                bs_train, bs_valid, bs_test, tokenizer=None):
#     # HACK: tokenizer as kwarg.
#     collate = lm_collate_factory(tokenizer)

#     train_dl = get_lm_train_dl(ds_train, bs_train, collate_fn=collate)
#     valid_dl = get_lm_eval_dl(ds_valid, bs_valid, collate_fn=collate)
#     test_dl = get_lm_eval_dl(ds_test, bs_test, collate_fn=collate)

#     return train_dl, valid_dl, test_dl

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
# Simpplified. dataset by name.
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


def new_distributed_get_train_test_dl_from_args(args, **kw):

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
# get x seperate from y, both with same seed
#############################################
# TODO: for masked/normal LM.

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
                                           dataset_keywords=dict(),
                                           **kw):

    experiment_manual_seed = torch.initial_seed()

    DICT_DATASET_JUST_XY_FUNC = {
        'cifar10': get_cifar_10_just_x_or_y_train_test_ds,
        'cifar100': get_cifar_100_just_x_or_y_train_test_ds,
        'imagenet': get_imagenet_just_x_or_y_train_test_ds
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
