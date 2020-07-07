# See https://huggingface.co/models
# GPT2_NAMES_OR_PATHES = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
# TODO: replace with auto config & tokenizers
from transformers import (GPT2Config, GPT2Tokenizer)
from transformers import (BertConfig, BertTokenizer)
from transformers import (AutoConfig, AutoTokenizer)

from .normal.NLP_models import (GPT2LMHeadModel, GPT2Model,
                                StatelessGPT2LMHeadModel)
from .normal.NLP_models import StatelessGPT2LMHeadModel
from .normal.NLP_models.modeling_bert_old import BertForQuestionAnswering
# from .normal.NLP_models.modeling_roberta import RobertaForQuestionAnswering
from .normal.NLP_models.modeling_roberta import RobertaForSequenceClassification
from .normal.NLP_models.modeling_bert import BertForSequenceClassification

import sys
from inspect import getmembers, isfunction

# We use this to get cfg_class, model_class, and tokenizer_class
MODEL_TYPES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'gpt2_lm': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2_lm_stateless': (GPT2Config, StatelessGPT2LMHeadModel, GPT2Tokenizer),
    'bert_squad_old': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'bert_glue': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta_glue':
    (AutoConfig, RobertaForSequenceClassification, AutoTokenizer),
}

# NOTE: some of these configs are just for this repo, see
# `models.transformers_utils.pretrained_model_config_and_tokenizer`


def roberta_large_8p_bw11_0_mnli_glue():
    return dict(model_type='roberta_glue',
                model_name_or_path='roberta-large',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                num_labels=3,
                finetuning_task='mnli')


# NOTE: this is to reprocduce a pretrained checkpoint on MNLI.
# def roberta_large_8p_bw11_0_mnli_glue():
#     return dict(model_type='roberta_glue',
#                 model_name_or_path='roberta-large-mnli',
#                 do_lower_case=False,
#                 output_past=False,
#                 stateless_tied=False,
#                 num_labels=3,
#                 finetuning_task='mnli')


def bert_large_uncased_whole_word_masking_8p_bw11_0_async_mnli_glue():
    return dict(model_type='bert_glue',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                num_labels=3,
                finetuning_task='mnli')


def gpt2_p4_lm_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False)


def gpt2_p4_lm_tied():
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


def gpt2_xl_p8_lm_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False)


def gpt2_xl_p8_lm_tied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                output_past=False,
                stateless_tied=True)


def bert_large_uncased_squad_8p():
    return dict(model_type='bert_squad_old',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False)


# GPipe version for the functions, as it has different balance.
def gpt2_p4_lm_tied_gpipe():
    return gpt2_p4_lm_tied()


def gpt2_p4_lm_untied_gpipe():
    return gpt2_p4_lm_untied()


def gpt2_xl_p8_lm_tied_gpipe():
    return gpt2_xl_p8_lm_tied()


def gpt2_xl_p8_lm_untied_gpipe():
    return gpt2_xl_p8_lm_untied()


#####################
# T5
####################
def t5_small_p4():
    return dict(model_type='t5',
                model_name_or_path='t5-small',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False)


def t5_small_p3_squad1():
    return dict(model_type='t5',
                model_name_or_path='t5-small',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "output_hidden_states": False
                },
                stateless_tied=False)


functions_list = getmembers(
    sys.modules[__name__],
    lambda o: isfunction(o) and o.__module__ == __name__)

# dict of all functions in current file.
# name --> function
MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS = {i: v for i, v in functions_list}
