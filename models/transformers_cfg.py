# See https://huggingface.co/models
# GPT2_NAMES_OR_PATHES = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
# TODO: replace with auto config & tokenizers
from transformers import (GPT2Config, GPT2Tokenizer)
from .normal import GPT2LMHeadModel, GPT2Model, StatelessGPT2LMHeadModel

# We use this to get cfg_class, model_class, and tokenizer_class
MODEL_TYPES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'gpt2_lm': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2_lm_stateless': (GPT2Config, StatelessGPT2LMHeadModel, GPT2Tokenizer),
}

# NOTE: some of these configs are just for this repo, see
# `models.transformers_utils.pretrained_model_config_and_tokenizer`

def gpt2_lowercase():
    return dict(model_type='gpt2',
                model_name_or_path='gpt2',
                do_lower_case=True,
                output_past=False)


def gpt2_lm_lowercase():
    return dict(model_type='gpt2_lm',
                model_name_or_path='gpt2',
                do_lower_case=True,
                output_past=False)


def gpt2xl_lm_lowercase():
    return dict(model_type='gpt2_lm',
                model_name_or_path='gpt2-xl',
                do_lower_case=True,
                output_past=False)


def gpt2_lmhead_lowercase_5p():
    """ Stateless version, were partitions 4 and 0 are on same GPU but different ranks
        Used for the tied weights trick.
     """
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2',
                do_lower_case=True,
                output_past=False,
                stateless_tied=True)


MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS = {
    'gpt2': gpt2_lowercase,
    'gpt2_lowercase': gpt2_lowercase,
    'gpt2_lm_lowercase': gpt2_lm_lowercase,
    'gpt2_lmhead_lowercase_5p': gpt2_lmhead_lowercase_5p,
}
