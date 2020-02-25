import torch
from typing import Tuple
from transformers import PreTrainedTokenizer

from .interface import DLTask
import types

# NOTE: Not much change from CV task, maybe this could just be removed
class LMTask(DLTask):
    def __init__(self, device, is_last_partition, is_first_partition):
        super().__init__()
        self.device = device

        # Determine unpack_cls
        if is_last_partition:
            # Last partition
            def unpack_cls(self, x):
                assert isinstance(x, tuple) or isinstance(x, list)
                return x,  # Comma here is important!
        elif is_first_partition:
            # Fist partition
            # NOTE: in masked LM we also mask...
            def unpack_cls(self, x):
                with torch.no_grad():
                    x = x.to(device, non_blocking=True)
                return (x, )
        else:
            # Mid partition
            def unpack_cls(self, x):
                assert isinstance(x, tuple) or isinstance(x, list)
                return x,

        # TODO: can be static...
        # types.MethodType
        self.unpack_data_for_partition = types.MethodType(unpack_cls, self)

    def unpack_data_for_partition(self, data):
        raise NotImplementedError()  # patched at init.

    def pack_send_context(self, model_out, *ctx):
        # ctx here is just the label y
        return (*model_out, *ctx)

    def preload_last_partition(self, dlitr, device):
        y = next(dlitr)
        return (y.to(device, non_blocking=True), )


# TODO: allow seperate, remove clone, etc...
# TODO: this should actually be moved to dataset! so the dataloader will handle all this mess.
# This is like a "transform"
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
# commitid: f54a5bd37f99e3933a396836cb0be0b5a497c077
def mask_tokens(inputs: torch.Tensor,
                tokenizer: PreTrainedTokenizer,
                mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling:
     80% MASK, 10% random, 10% original.

     Usage:
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
     """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer.")

    # FIXME:
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
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape,
                                                  0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    # NOTE: (SE) shouldn't it be 0.1 instead of 0.5? its probobly happening, and I don't see it.
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer),
                                 labels.shape,
                                 dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
