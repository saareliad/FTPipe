# See https://huggingface.co/models
# GPT2_NAMES_OR_PATHES = {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
# TODO: replace with auto config & tokenizers
import sys
from inspect import getmembers, isfunction

from transformers import (BertConfig, BertTokenizer)
from transformers import (GPT2Config, GPT2Tokenizer)
from transformers import (RobertaConfig, RobertaTokenizer)
from transformers import (T5Config, T5Tokenizer)

from models.normal.NLP_models.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from models.normal.NLP_models.modeling_gpt2_tied_weights import GPT2LMHeadModel as StatelessGPT2LMHeadModel

from models.normal.NLP_models.modeling_bert import BertForQuestionAnswering as BertForQuestionAnsweringNew
from models.normal.NLP_models.modeling_bert import BertForSequenceClassification
from models.normal.NLP_models.modeling_bert_old import BertForQuestionAnswering
# from .normal.NLP_models.modeling_roberta import RobertaForQuestionAnswering
from models.normal.NLP_models.modeling_roberta import RobertaForSequenceClassification
from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration
from models.normal.NLP_models.modeling_t5_tied_weights import T5ForConditionalGeneration as StatelesT5ForConditionalGeneration

# We use this to get cfg_class, model_class, and tokenizer_class
MODEL_TYPES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'gpt2_lm': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2_lm_stateless': (GPT2Config, StatelessGPT2LMHeadModel, GPT2Tokenizer),
    'bert_squad_old': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'bert_squad': (BertConfig, BertForQuestionAnsweringNew, BertTokenizer),
    'bert_glue': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta_glue':
        (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    't5_stateless':
        (T5Config, StatelesT5ForConditionalGeneration, T5Tokenizer),
}


# NOTE: some of these configs are just for this repo, see
# `models.transformers_utils.pretrained_model_config_and_tokenizer`

def roberta_large_8p_bw11_0_async_mnli_glue():
    return dict(model_type='roberta_glue',
                # model_name_or_path='roberta-large-mnli',  # to reproduce a pretrained checkpoint on MNLI.
                model_name_or_path='roberta-large',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                explicitly_set_dict={
                    "precompute_attention_mask": True
                },
                num_labels=3,
                finetuning_task='mnli')


def bert_large_uncased_whole_word_masking_8p_bw11_0_async_rte_glue():
    return dict(model_type='bert_glue',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=False,
                output_past=False,
                explicitly_set_dict={
                    "precompute_attention_mask": True
                },
                stateless_tied=False,
                num_labels=2,
                finetuning_task='rte')


def bert_large_uncased_whole_word_masking_8p_bw11_0_async_mnli_glue():
    return dict(model_type='bert_glue',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                num_labels=3,
                explicitly_set_dict={
                    "precompute_attention_mask": True
                },
                finetuning_task='mnli')


def bert_base_uncased_4p_bw11_0_async_mnli_glue():
    return dict(model_type='bert_glue',
                model_name_or_path='bert-base-uncased',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                num_labels=3,
                finetuning_task='mnli')


def bert_base_uncased_8p_bw11_0_async_mnli_glue():
    return dict(model_type='bert_glue',
                model_name_or_path='bert-base-uncased',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                num_labels=3,
                explicitly_set_dict={
                    "precompute_attention_mask": True
                },
                finetuning_task='mnli')


def roberta_base_8p_bw11_0_async_mnli_glue():
    return dict(model_type='roberta_glue',
                model_name_or_path='roberta-base',
                do_lower_case=False,
                output_past=False,
                stateless_tied=False,
                num_labels=3,
                explicitly_set_dict={
                    "precompute_attention_mask": True
                },
                finetuning_task='mnli')


def gpt2_p4_lm_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2',
                do_lower_case=False,
                explicitly_set_dict=dict(
                    output_past=False, ),
                stateless_tied=False)


def gpt2_p4_lm_tied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2',
                do_lower_case=False,
                explicitly_set_dict=dict(
                    output_past=False, ),
                stateless_tied=True)


def new_gpt2_xl_tied_lm_p8_seq_512():
    return dict(model_type='gpt2_lm',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                explicitly_set_dict=dict(
                    output_past=False, ),
                stateless_tied=False)


def old_gpt2xl_8p_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                explicitly_set_dict=dict(
                    output_past=False, ),
                stateless_tied=False)


def gpt2_xl_p8_lm_untied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                explicitly_set_dict=dict(
                    output_past=False, ),
                stateless_tied=False)


def gpt2_xl_p8_lm_tied():
    return dict(model_type='gpt2_lm_stateless',
                model_name_or_path='gpt2-xl',
                do_lower_case=False,
                explicitly_set_dict=dict(
                    output_past=False, ),
                stateless_tied=True)


def bert_large_uncased_squad_8p():
    return dict(model_type='bert_squad_old',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False)

def bert_base_uncaseds_384_2p_bw12_pipedream():
    return dict(model_type='bert_squad_old',
                model_name_or_path='bert-base-uncased',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False,
                explicitly_set_dict={
                    'return_dict': False
                },
                do_resize_token_embedding=False,
                )

def bert_base_uncaseds_384_2p_bw12_async_pipedream():
    return dict(model_type='bert_squad_old',
                model_name_or_path='bert-base-uncased',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False,
                explicitly_set_dict={
                    'return_dict': False
                },
                do_resize_token_embedding=False,
                )

def bert_large_uncased_whole_word_maskings_384_2p_bw12_pipedream():
    return dict(model_type='bert_squad',
              model_name_or_path='bert-large-uncased-whole-word-masking',
              do_lower_case=True,
              output_past=False,
              stateless_tied=False,
              explicitly_set_dict={'precompute_attention_mask': True, 'return_dict': False},
              do_resize_token_embedding=False,
              )


def bert_large_uncased_whole_word_maskings_384_2p_bw12_async_pipedream():
    return dict(model_type='bert_squad',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False,
                explicitly_set_dict={'precompute_attention_mask': True, 'return_dict': False},
                do_resize_token_embedding=False,
                )


def bert_large_uncased_whole_word_maskings_384_8p_bw12_pipedream():
    return dict(model_type='bert_squad',
              model_name_or_path='bert-large-uncased-whole-word-masking',
              do_lower_case=True,
              output_past=False,
              stateless_tied=False,
              explicitly_set_dict={'precompute_attention_mask': True, 'return_dict': False},
              do_resize_token_embedding=False,
              )


def bert_large_uncased_whole_word_maskings_384_8p_bw12_async_pipedream():
    return dict(model_type='bert_squad',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False,
                explicitly_set_dict={'precompute_attention_mask': True, 'return_dict': False},
                do_resize_token_embedding=False,
                )


def bert_large_uncased_whole_word_maskings_384_4p_bw12_pipedream():
    return dict(model_type='bert_squad',
              model_name_or_path='bert-large-uncased-whole-word-masking',
              do_lower_case=True,
              output_past=False,
              stateless_tied=False,
              explicitly_set_dict={'precompute_attention_mask': True, 'return_dict': False},
              do_resize_token_embedding=False,
              )


def bert_large_uncased_whole_word_maskings_384_4p_bw12_async_pipedream():
    return dict(model_type='bert_squad',
                model_name_or_path='bert-large-uncased-whole-word-masking',
                do_lower_case=True,
                output_past=False,
                stateless_tied=False,
                explicitly_set_dict={'precompute_attention_mask': True, 'return_dict': False},
                do_resize_token_embedding=False,
                )


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

def t5_small_tied_lmhead_4p_bw12_async_squad1():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-small',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_large_tied_lmhead_8p_bw12_async_squad1():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-large',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_320_8_8p_bw12_squad1():
    # also, seq sends, mem constaint, ....
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_64_6_8p_bw12_squad1_virtual_stages():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)

def t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)

def t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)

def t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_L32():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_512_4_8p_bw12_squad1_acyclic():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_small_tied_lmheads_512_4_3p_bw12_squad1_virtual_stages():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-small',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)


def t5_3b_tied_lmheads_64_4_8p_bw12_squad1():
    return dict(model_type='t5_stateless',
                model_name_or_path='t5-3b',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                explicitly_set_dict={
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": True,
                    "output_hidden_states": False
                },
                stateless_tied=True)



functions_list = getmembers(
    sys.modules[__name__],
    lambda o: isfunction(o) and o.__module__ == __name__)

# dict of all functions in current file.
# name --> function
MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS = {i: v for i, v in functions_list}


def _register_hardcoded(name, dict):
    global MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS
    MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS[name] = dict

# for name in MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS:
#     # HACK: called with a ready model instance.
#     register_model(name=name, dict_params={}, model_class=None, generated_file_name_or_path=name)
