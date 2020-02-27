from .cfg_to_model import get_partitioning
import importlib
from transformers import (GPT2Config, GPT2Tokenizer)

# FIXME: model with LM head is actually not good for pipeline,
# as first partition does not care about lables.
from .normal import GPT2LMHeadModel, GPT2Model

MODEL_TYPES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'gpt2_lm': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

# See https://huggingface.co/models
# GPT2_NAMES_OR_PATHES = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}


def resize_token_embeddings(model, tokenizer):
    # NOTE: must use the same tokenizer used at partitioning.
    model_to_resize = model.module if hasattr(model, 'module') else model
    model_to_resize.resize_token_embeddings(len(tokenizer))


def get_block_size(tokenizer, block_size=-1):
    if block_size <= 0:
        # Our input block size will be the max possible for the model
        block_size = tokenizer.max_len_single_sentence
    block_size = min(block_size, tokenizer.max_len_single_sentence)
    return block_size


def pretrained_model_config_and_tokenizer(
        model_type: str = 'gpt2',
        config_name: str = "",
        model_name_or_path: str = 'gpt2',
        tokenizer_name: str = "",
        do_lower_case: bool = True,
        cache_dir: str = "",
):

    config_class, model_class, tokenizer_class = MODEL_TYPES[model_type]
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir if cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir if cache_dir else None)

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool('.ckpt' in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None)

    resize_token_embeddings(model, tokenizer)

    return model, tokenizer, config


def gpt2_lowercase():
    mycfg = dict(model_type='gpt2',
                 model_name_or_path='gpt2',
                 do_lower_case=True)
    model, tokenizer, config = pretrained_model_config_and_tokenizer(**mycfg)
    resize_token_embeddings(model, tokenizer)

    return model, tokenizer, config


def gpt2_lowecase_partitioning():
    model, tokenizer, config = gpt2_lowercase()
    partitioning_cfg = get_partitioning('gpt2', model_instance=model)
    # NOTE: can use tokenizer.max_len_single_sentence to get maximum allowed block size.
    return partitioning_cfg, tokenizer, config


