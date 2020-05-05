# See https://huggingface.co/models
# GPT2_NAMES_OR_PATHES = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
# TODO: replace with auto config & tokenizers
from transformers import (GPT2Config, GPT2Tokenizer)
from .normal import GPT2LMHeadModel, GPT2Model, StatelessGPT2LMHeadModel

import sys
from inspect import getmembers, isfunction

# We use this to get cfg_class, model_class, and tokenizer_class
MODEL_TYPES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'gpt2_lm': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2_lm_stateless': (GPT2Config, StatelessGPT2LMHeadModel, GPT2Tokenizer),
}

# NOTE: some of these configs are just for this repo, see
# `models.transformers_utils.pretrained_model_config_and_tokenizer`


def gpt2_p4_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False)


def gpt2_p4_tied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2',
                do_lower_case=False,
                output_past=False,
                stateless_tied=True)


def old_gpt2xl_8p_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False)


def gpt2xl_p8_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False)


def gpt2xl_p8_tied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                output_past=False,
                stateless_tied=True)


# GPipe version for the functions, as it has different balance.
def gpt2_p4_tied_gpipe():
    return gpt2_4p_tied()


def gpt2_p4_untied_gpipe():
    return gpt2_4p_untied()


def gpt2xl_p8_tied_gpipe():
    return gpt2xl_8p_tied()


def gpt2xl_p8_untied_gpipe():
    return gpt2xl_p8_untied()


functions_list = getmembers(
    sys.modules[__name__],
    lambda o: isfunction(o) and o.__module__ == __name__)

# dict of all functions in current file.
# name --> function
MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS = {i: v for i, v in functions_list}
