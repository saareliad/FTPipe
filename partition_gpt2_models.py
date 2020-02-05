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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config,
                          #    GPT2LMHeadModel,
                          GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          #   CamembertConfig, CamembertForMaskedLM, CamembertTokenizer
                          )

from sample_models import GPT2LMHeadModel

from pytorch_Gpipe import pipe_model
from misc import run_analysis, run_partitions
from pytorch_Gpipe.model_profiling import Node, NodeTypes

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    # 'camembert': (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)
}


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path), file_path
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_name_or_path + '_cached_lm_' + str(block_size) + '_' + filename)

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
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i:i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(
        tokenizer, args, file_path=args.train_data_file, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(
        val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(
        special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def node_weight_function(node: Node):
    if node.type is NodeTypes.LAYER:
        return int(node.weight.forward_time + node.weight.backward_time)
    if node.type is NodeTypes.CONSTANT:
        return 0
    if node.type is NodeTypes.OP:
        return 1
    return 0


def edge_weight_function(bandwidth_gps):
    def f(u: Node, v: Node):
        if u.type is NodeTypes.LAYER:
            return max(1, int(u.weight.output_size / bandwidth_gps))
        if v.type is NodeTypes.LAYER:
            return max(1, int(v.weight.input_size / bandwidth_gps))
        if u.type is NodeTypes.CONSTANT:
            return 1000
        return 1
    return f


def partition_model(args, train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.partitioning_batch_size)

    model_to_resize = model.module if hasattr(model, 'module') else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    batch = next(iter(train_dataloader))

    inputs, labels = mask_tokens(
        batch, tokenizer, args) if args.mlm else (batch, batch)
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
    sample = (inputs, labels)
    model.train()
    graph = pipe_model(model, sample, depth=args.depth, n_iter=args.n_iter, nparts=args.n_partitions,
                       node_weight_function=node_weight_function,
                       edge_weight_function=edge_weight_function(
                           args.bandwidth_gps),
                       use_layers_only_graph=args.partition_layer_graph,
                       output_file=args.output_file, DEBUG=False)
    graph.save_as_pdf(args.output_file, ".")

    generated = importlib.import_module(args.output_file)
    create_pipeline_configuration = generated.create_pipeline_configuration

    GET_PARTITIONS_ON_CPU = True

    config = create_pipeline_configuration(model, partitions_only=False,
                                           DEBUG=GET_PARTITIONS_ON_CPU)

    # out = run_partitions(sample, config)
    bandwidth_gps = args.bandwidth_gps
    recomputation = not args.no_recomputation

    # _ = run_partitions(sample, config)
    if not args.no_analysis:
        # TODO assumes batch is first
        shape = (args.analysis_batch_size, inputs.shape[1])
        expanded_inputs = inputs[0].unsqueeze(0).expand(shape)
        shape = (args.analysis_batch_size, labels.shape[1])
        expanded_labels = labels[0].unsqueeze(0).expand(shape)

        analysis_sample = (expanded_inputs, expanded_labels)

        run_analysis(analysis_sample, graph, config, args.n_iter,
                     recomputation=recomputation, bandwidth_gps=bandwidth_gps)

    # model(inputs)
    # outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(
    #     inputs, labels=labels)
    # # model outputs are always tuple in transformers (see doc)
    # loss = outputs[0]


def main():
    parser = argparse.ArgumentParser()

    # parameter required by the repo
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="gpt2", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # parameters of the partitioning script
    parser.add_argument('--partitioning_batch_size', type=int, default=1,
                        help="batch size to use when partitioning the model")
    parser.add_argument('--model_too_big', action='store_true', default=False,
                        help="if the model is too big run the whole partitioning process on CPU, and drink a cup of coffee in the meantime")
    parser.add_argument('--n_partitions', type=int, default=4)
    parser.add_argument('--output_file', default='gpt2')
    parser.add_argument('--auto_file_name', action='store_true',
                        default=False, help="create file name automatically")
    parser.add_argument('--n_iter', type=int, default=10,
                        help="number of iteration used in order to profile the network and run analysis")
    parser.add_argument('--bandwidth_gps', type=float,
                        default=12, help="data transfer rate between gpus in gigabaytes per second")
    parser.add_argument('--no_recomputation', action='store_true',
                        default=False, help="whether to use recomputation for the backward pass")
    parser.add_argument('--analysis_batch_size', type=int, default=1,
                        help="batch size to use when analysing the generated partititon")
    parser.add_argument('--no_analysis', action='store_true',
                        default=False, help="disable partition analysis")
    parser.add_argument("--depth", default=1000, type=int,
                        help="the depth in which we will partition the model")
    parser.add_argument("--partition_layer_graph", action="store_true",
                        default=True, help="whether to partition a graph containing only layers")

    args = parser.parse_args()

    if args.auto_file_name:
        args.output_file = f"{args.model_type}_p{args.n_partitions}"

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    # Setup CUDA, GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.model_too_big else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(
                                            '.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer)

    partition_model(args, train_dataset, model, tokenizer)

# download dataset from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip

# export TRAIN_FILE=wikitext-2-raw/wiki.train.raw


# python partition_gpt2_models.py --train_data_file=$TRAIN_FILE
if __name__ == "__main__":
    main()
