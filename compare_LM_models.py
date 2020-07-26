# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import pickle
import random
import importlib
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import warnings
import sys
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    GPT2Config,
    #    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    #   CamembertConfig, CamembertForMaskedLM, CamembertTokenizer
)
import itertools
from models.normal import GPT2LMHeadModel, GPT2Model
from models.normal import StatelessGPT2LMHeadModel  # , StatelessGPT2Model


from misc import run_analysis, run_partitions
from misc.analysis_utils import convert_to_analysis_format

MODEL_CLASSES_LM_HEAD = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert':
    (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    # 'camembert': (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)
}

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
}

MODEL_CLASSES_LM_HEAD_STATELESS_TIED = {
    'gpt2': (GPT2Config, StatelessGPT2LMHeadModel, GPT2Tokenizer),  # TODO:
}


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path), file_path
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_name_or_path + '_cached_lm_' +
            str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)

        else:
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
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


def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(tokenizer,
                          args,
                          file_path=args.train_data_file,
                          block_size=args.block_size)
    return dataset



def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                 dtype=torch.bool),
                                    value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape,
                                                  0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer),
                                 labels.shape,
                                 dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def compare_models(args,
                   train_dataset,
                   our_baseline_model,
                   huggingface_model,
                   tokenizer):
    ourbase_path=inspect.getmodule(our_baseline_model).__name__+"."+type(our_baseline_model).__name__
    huggingface_path =inspect.getmodule(huggingface_model).__name__+"."+type(huggingface_model).__name__
    tokenizer_path=inspect.getmodule(tokenizer).__name__+"."+type(tokenizer).__name__
    partitioned_path = args.partitioned_model_path
    print(f"comparing NLP models")
    print(f"our model: {ourbase_path}")
    print(f"partitioned model: {partitioned_path}")
    print(f"huggingface reference: {huggingface_path}")
    print(f"tokenizer: {tokenizer_path}")
 
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.comparison_batch_size)

    # prepare models
    huggingface_model.resize_token_embeddings(len(tokenizer))

    our_baseline_model.resize_token_embeddings(len(tokenizer))
    our_baseline_model.make_stateless_after_loaded_tied_and_resized()

    #select a random batch
    batch_idx = np.random.randint(0,high=len(train_dataloader),size=1)[0]
    print("selected batch index:",batch_idx)
    batch=next(itertools.islice(train_dataloader, batch_idx, None))

    inputs, labels = mask_tokens(batch, tokenizer,
                                 args) if args.mlm else (batch, batch)
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
    sample = (inputs, labels)

    generated = importlib.import_module(args.partitioned_model_path)
    layerDict = generated.layerDict
    tensorDict=generated.tensorDict
    create_pipeline_configuration = generated.create_pipeline_configuration

    GET_PARTITIONS_ON_CPU = True
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

    print()

    ######################
    print("comparing our model to partitioned")
    # ####################
    depth = args.depth
    blocks = args.basic_blocks
    analysis_config = convert_to_analysis_format(config,
        layerDict(our_baseline_model, depth=depth, basic_blocks=blocks),
        tensorDict(our_baseline_model))
    print("comparing train")
    with torch.no_grad():
        our_baseline_model = our_baseline_model.cuda().train()
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        expected = our_baseline_model(*sample)[0]
        our_baseline_model.cpu()

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        actual = run_partitions(sample, analysis_config)[0]
        assert torch.allclose(expected, actual),"partitioned != baseline"
    print("our model is identical to our partitioned during training")

    print("comparing eval")
    with torch.no_grad():
        our_baseline_model = our_baseline_model.cuda().eval()
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        expected = our_baseline_model(*sample)[0]
        our_baseline_model=our_baseline_model.cpu()

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        actual = run_partitions(sample, analysis_config)[0]
        assert torch.allclose(expected, actual),"partitioned != baseline"
    print("our model is identical to our partitioned during evaluation")
    our_baseline_model=our_baseline_model.cpu()

    print()

    ######################
    print("comparing huggingface to our model")
    # ####################
    print("comparing train")
    with torch.no_grad():
        huggingface_model = huggingface_model.cuda().train()
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        expected = huggingface_model(*sample)[0]
        huggingface_model=huggingface_model.cpu()

        our_baseline_model = our_baseline_model.cuda().train()
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        actual = our_baseline_model(*sample)[0]
        assert torch.allclose(expected, actual),"our model != huggingface"
        our_baseline_model=our_baseline_model.cpu()
    
    print("our model is identical to huggingface reference during training")
    print("comparing eval")
    with torch.no_grad():
        huggingface_model = huggingface_model.cuda().eval()
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        expected = huggingface_model(*sample)[0]
        huggingface_model=huggingface_model.cpu()

        our_baseline_model = our_baseline_model.cuda().eval()
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        actual = our_baseline_model(*sample)[0]
        assert torch.allclose(expected, actual),"our model != huggingface"
        our_baseline_model = our_baseline_model.cpu()
    print("our model is identical to huggingface reference during evaluation")


    ######################
    print("\ncomparing gradient huggingface to our model")
    # ####################

    huggingface_ps = {n for n,p in huggingface_model.named_parameters()}
    our_ps = {n for n,p in our_baseline_model.named_parameters()}
    #names are not shared so we cannot compare in general
    ignored_ps =huggingface_ps.symmetric_difference(our_ps)
    print(f"the following gradients are ignored:\n{ignored_ps}")    

    huggingface_model = huggingface_model.cuda().train()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    huggingface_model(*sample)[0].backward()
    huggingface_model = huggingface_model.cpu()

    our_baseline_model = our_baseline_model.cuda().train()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    our_baseline_model(*sample)[0].backward()
    our_baseline_model = our_baseline_model.cpu()
    actual_ps = {n:p for n,p in our_baseline_model.named_parameters()}

    for n,p in huggingface_model.named_parameters():
        if n in ignored_ps:
            continue
        assert torch.allclose(p.grad,actual_ps[n].grad),f"grad {n} is not the same"
    
    #TODO hardcoded for gpt2 tied weights
    assert hasattr(our_baseline_model,"w_wte")
    print("explicitly comparing w_wte to transformer.wte.weight")
    huggingface_wte=huggingface_model.transformer.wte.weight
    our_wte = our_baseline_model.w_wte
    assert torch.allclose(huggingface_wte.grad,our_wte.grad),"the shared weight grad is not the same"
    print("gradients are the same")
    
    #reset grads
    for p in itertools.chain(huggingface_model.parameters(),our_baseline_model.parameters()):
        p.grad=None


    ######################
    print("\ncomparing gradient huggingface to our partitioned")
    print(f"the following gradients are ignored:\n{ignored_ps}")  
    # ####################
    huggingface_model = huggingface_model.cuda().train()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    huggingface_model(*sample)[0].backward()
    huggingface_model = huggingface_model.cpu()

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    run_partitions(sample, analysis_config)[0].backward()
    our_baseline_model = our_baseline_model.cpu()
    actual_ps = {n:p for n,p in our_baseline_model.named_parameters()}

    for n,p in huggingface_model.named_parameters():
        if n in ignored_ps:
            continue
        assert torch.allclose(p.grad,actual_ps[n].grad),f"grad {n} is not the same"
    
    #TODO hardcoded for gpt2 tied weights
    assert hasattr(our_baseline_model,"w_wte")
    print("explicitly comparing w_wte to transformer.wte.weight")
    huggingface_wte=huggingface_model.transformer.wte.weight
    our_wte = our_baseline_model.w_wte
    assert torch.allclose(huggingface_wte.grad,our_wte.grad),"the shared weight grad is not the same"
    print("gradients are the same")
    
    #reset grads
    for p in itertools.chain(huggingface_model.parameters(),our_baseline_model.parameters()):
        p.grad=None

    


def parse_cli():
    # And add extra args spesific to this script as needed.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parameter required by the repo
    tr_params = parser.add_argument_group("Transformers parameters")
    tr_params.add_argument("--train_data_file",
                           default=None,
                           type=str,
                           required=True,
                           help="The input training data file (a text file).")
    tr_params.add_argument("--model_type",
                           default="gpt2",
                           type=str,
                           help="The model architecture to be fine-tuned.")
    tr_params.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="The model checkpoint for weights initialization.")
    tr_params.add_argument(
        "--mlm",
        action='store_true',
        help="Train with masked-language modeling loss instead of language modeling."
    )
    tr_params.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss")
    tr_params.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path"
    )
    tr_params.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path"
    )
    tr_params.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)"
    )
    tr_params.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens)."
    )
    tr_params.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    tr_params.add_argument(
        '--overwrite_cache',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    tr_params.add_argument('--seed',
                           type=int,
                           default=42,
                           help="random seed for initialization")

    tr_params.add_argument("--lmhead",
                           default=False,
                           action="store_true",
                           help="Partition a model with LM head")

    tr_params.add_argument(
        "--stateless_tied",
        default=False,
        action="store_true",
        help="Tie weights stateless trick. Note that shared weight may be sent in pipe"
    )

    comp_params = parser.add_argument_group("comparison parameters")
    comp_params.add_argument("--partitioned_model_path",
    required=True,
    type=str,
    default='gpt2')

    # parameters of the partitioning script
    comp_params.add_argument('--comparison_batch_size',
                        type=int,
                        required=True,
                        help="batch size to use when partitioning the model")

    args = parser.parse_args()

    return args


def get_model(args, stateless):
    if stateless:
        model_class_dict_to_use = MODEL_CLASSES_LM_HEAD_STATELESS_TIED
    else:
        model_class_dict_to_use = MODEL_CLASSES_LM_HEAD

    config_class, model_class, tokenizer_class = model_class_dict_to_use[
        args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None).cpu()

    return model


def get_tokenizer(args,stateless):
    if stateless:
        model_class_dict_to_use = MODEL_CLASSES_LM_HEAD_STATELESS_TIED
    else:
        model_class_dict_to_use = MODEL_CLASSES_LM_HEAD

    config_class, model_class, tokenizer_class = model_class_dict_to_use[
        args.model_type]
    
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    return tokenizer

def main():

    args = parse_cli()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"
                           ] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling).")

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = get_tokenizer(args,args.stateless_tied)

    train_dataset = load_and_cache_examples(args, tokenizer)

    compare_models(args,
                   train_dataset,
                   get_model(args, args.stateless_tied),
                   get_model(args, False),
                   tokenizer,
                   )


if __name__ == "__main__":

    main()
