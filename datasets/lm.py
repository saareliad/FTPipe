
import os
import torch
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
import pickle
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from .hardcoded_dirs import DEFAULT_DATA_DIR


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
                # NOTE: this makes it is suitable mostly for small datasets...
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
            # If your dataset is small, first you should look for a bigger one :-) and second you
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


# NOTE: This is like a "transform", Should be used straight in the dataset,
# so the dataloader will handle this.
# NOTE: we also provide 2 more functions just for inputs/labels
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
# commit_id: f54a5bd37f99e3933a396836cb0be0b5a497c077
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


def get_wikitext2_raw_train_test_ds(model_name_or_path,
                                    tokenizer,
                                    train_seq_len=512,
                                    test_seq_len=512,
                                    overwrite_cache=False,
                                    DATA_DIR=DEFAULT_DATA_DIR):
    """ Returns train and test datasets """

    train_ds = get_wikitext2_raw_train_valid_test_ds(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        split='train',
        block_size=train_seq_len,
        overwrite_cache=overwrite_cache)
    test_ds = get_wikitext2_raw_train_valid_test_ds(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        split='test',
        block_size=test_seq_len,
        overwrite_cache=overwrite_cache)
    return train_ds, test_ds


def get_wikitext2_raw_train_valid_ds(model_name_or_path,
                                     tokenizer,
                                     train_seq_len=512,
                                     valid_seq_len=512,
                                     overwrite_cache=False,
                                     DATA_DIR=DEFAULT_DATA_DIR):

    train_ds = get_wikitext2_raw_train_valid_test_ds(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        split='train',
        block_size=train_seq_len,
        overwrite_cache=overwrite_cache)
    valid_ds = get_wikitext2_raw_train_valid_test_ds(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        split='valid',
        block_size=valid_seq_len,
        overwrite_cache=overwrite_cache)
    return train_ds, valid_ds


def get_wikitext2_raw_test_ds(model_name_or_path,
                              tokenizer,
                              test_seq_len=512,
                              overwrite_cache=False,
                              DATA_DIR=DEFAULT_DATA_DIR):
    test_ds = get_wikitext2_raw_train_valid_test_ds(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        split='test',
        block_size=test_seq_len,
        overwrite_cache=overwrite_cache)
    return test_ds


################
# Get DL
################


def lm_collate_factory(tokenizer):
    assert tokenizer is not None

    def lm_collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples,
                            batch_first=True,
                            padding_value=tokenizer.pad_token_id)

    return lm_collate


def get_lm_train_dl(ds_train,
                    bs_train,
                    tokenizer=None,
                    collate_fn=None,
                    shuffle=True,
                    **kw):
    collate = collate_fn if collate_fn else lm_collate_factory(tokenizer)
    train_sampler = RandomSampler(ds_train)
    train_dl = DataLoader(ds_train,
                          shuffle=False,
                          sampler=train_sampler,
                          batch_size=bs_train,
                          collate_fn=collate,
                          **kw)
    return train_dl


def get_lm_eval_dl(ds_eval,
                   bs_eval,
                   tokenizer=None,
                   shuffle=False,
                   collate_fn=None,
                   **kw):
    collate = collate_fn if collate_fn else lm_collate_factory(tokenizer)
    eval_sampler = SequentialSampler(ds_eval)
    eval_dl = DataLoader(bs_eval,
                         sampler=eval_sampler,
                         batch_size=bs_eval,
                         shuffle=False,
                         collate_fn=collate,
                         **kw)
    return eval_dl


def get_lm_train_valid_dl(ds_train,
                          ds_test,
                          bs_train,
                          bs_test,
                          tokenizer=None,
                          **kw):
    # HACK: tokenizer as kwarg.
    # HACK: parameters names are 'test' for backward compatability.
    if 'collate_fn' not in kw:
        collate = lm_collate_factory(tokenizer)
        kw['collate_fn'] = collate

    train_dl = get_lm_train_dl(ds_train, bs_train, **kw)
    valid_dl = get_lm_eval_dl(ds_test, bs_test, **kw)

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


#############################################
# get x separate from y, both with same seed
#############################################


def get_wt2_just_x_or_y_train_valid_ds(just, DATA_DIR=DEFAULT_DATA_DIR, **kw):
    # we don't use the just. its the same for all.
    return get_wikitext2_raw_train_valid_ds(DATA_DIR=DATA_DIR, **kw)


def get_wt2_just_x_or_y_train_test_ds(just, DATA_DIR=DEFAULT_DATA_DIR, **kw):
    # we don't use the just. its the same for all.
    return get_wikitext2_raw_train_test_ds(DATA_DIR=DATA_DIR, **kw)


def get_wt2_just_x_or_y_test_ds(just, DATA_DIR=DEFAULT_DATA_DIR, **kw):
    return get_wikitext2_raw_test_ds(DATA_DIR=DATA_DIR, **kw)