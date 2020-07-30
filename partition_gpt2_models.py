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
Fine-tuning the library models for language modeling on a text file (GPT-2, CTRL).
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
import warnings
import sys
from partition_scripts_utils import ParseMetisOpts,ParseAcyclicPartitionerOpts, ParsePartitioningOpts, record_cmdline,choose_blocks
from heuristics import get_weight_functions
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    CTRLConfig,
    CTRLTokenizer,
)

from models.normal import GPT2LMHeadModel, GPT2Model
from models.normal import StatelessGPT2LMHeadModel
from models.normal import CTRLLMHeadModel, CTRLModel
from models.normal import StatelessCTRLLMHeadModel

from pytorch_Gpipe import pipe_model
from misc import run_analysis,convert_to_analysis_format
from pytorch_Gpipe.utils import layerDict, tensorDict
import functools
from partition_async_pipe import partition_async_pipe
import math
from pytorch_Gpipe.model_profiling.tracer import register_new_traced_function,register_new_explicit_untraced_function
import operator

MODEL_CLASSES_LM_HEAD = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer)
}

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'ctrl': (CTRLConfig, CTRLModel, CTRLTokenizer)
}

MODEL_CLASSES_LM_HEAD_STATELESS_TIED = {
    'gpt2': (GPT2Config, StatelessGPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLConfig, StatelessCTRLLMHeadModel, CTRLTokenizer)
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

            tokenized_text = tokenizer.tokenize(text)
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenized_text)

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def partition_model(args,
                    train_dataset,
                    model,
                    tokenizer,
                    lm=True):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.partitioning_batch_size)

    model_to_resize = model.module if hasattr(model, 'module') else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    # Tie weights artificially using statless trick
    if args.stateless_tied:
        model_to_resize.make_stateless_after_loaded_tied_and_resized()

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    batch = next(iter(train_dataloader))

    inputs = batch.to(args.device)
    
    if lm:
        labels = batch.to(args.device)
        sample = (inputs, labels)
    else:
        sample = inputs

    model.train()
    batch_dim = 0
    bwd_to_fwd_ratio = args.bwd_to_fwd_ratio

    recomputation = not args.no_recomputation

    # so we could trace math.sqrt in gpt2 attention
    register_new_traced_function(math.sqrt, namespace=math)
    register_new_explicit_untraced_function(operator.is_,
                                            namespace=operator)
    register_new_explicit_untraced_function(operator.is_not,
                                            namespace=operator)
    
    def force_no_recomputation_fn(scope):
        if "stateless_lm_head" in scope or "lm_head" in scope:
            return True
    args.basic_blocks = choose_blocks(model,args)
    bw = args.bw

    node_weight_function, edge_weight_function = get_weight_functions(args, verbose=True)
    
    partial_pipe_model = functools.partial(
        pipe_model,
        model,
        batch_dim,
        sample,
        basic_blocks=args.basic_blocks,
        depth=args.depth,
        n_iter=args.n_iter,
        nparts=args.n_partitions,
        node_weight_function=node_weight_function,
        edge_weight_function=edge_weight_function,
        use_layers_only_graph=True,  # FIXME:
        use_graph_profiler=not args.use_network_profiler,
        use_network_profiler=args.use_network_profiler,
        profile_ops=not args.disable_op_profiling,
        output_file=args.output_file,
        generate_model_parallel=args.generate_model_parallel,
        generate_explicit_del=args.generate_explicit_del,
        save_memory_mode=args.save_memory_mode,
        recomputation=recomputation,
        use_METIS=args.use_METIS,
        acyclic_opt=args.acyclic_opt,
        METIS_opt=args.METIS_opt)

    if args.async_pipeline and (not args.no_recomputation):
        print("using async partitioner")
        graph=partition_async_pipe(args,model,0,sample,
                                    node_weight_function=node_weight_function,
                                    edge_weight_function=edge_weight_function,)
    else:
        graph = partial_pipe_model(
            force_no_recomp_scopes=force_no_recomputation_fn)

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")
        graph.serialize(args.output_file)

    # Replace the dummy partition wtih cuda:0.
    if args.stateless_tied:
        try:
            import subprocess
            subprocess.check_output([
                'sed', '-s', '-i', f"s/cuda:{args.n_partitions}/cuda:0/g",
                args.output_file + ".py"
            ])
        except:
            print("Failed to replaced tied dummy partition device")

    record_cmdline(args.output_file)    
    
    #record model creation args
    with open(f"{args.output_file}.py", "a") as f:
        model_args = {"model_type":args.model_type,
                     "model_name_or_path":args.model_name_or_path,
                     "lmhead":args.lmhead,
                     "stateless_tied":args.stateless_tied,
                     "do_lower_case":args.do_lower_case,
                     "block_size":args.block_size
                    }
        f.write("\n")
        f.write(f"model_args = {model_args}")
    

    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)

    create_pipeline_configuration = generated.create_pipeline_configuration

    GET_PARTITIONS_ON_CPU = True

    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

    bw = args.bw

    if not args.no_analysis:
        depth = args.depth
        blocks = args.basic_blocks
        analysis_config = convert_to_analysis_format(config,
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

        shape = (args.analysis_batch_size, inputs.shape[1])
        expanded_inputs = inputs[0].unsqueeze(0).expand(shape)

        if lm: 
            shape = (args.analysis_batch_size, labels.shape[1])
            expanded_labels = labels[0].unsqueeze(0).expand(shape)
            analysis_sample = (expanded_inputs, expanded_labels)
        else:
            analysis_sample = expanded_inputs
        
        stages_on_same_gpu = set()
        if args.lmhead and args.stateless_tied and len(
                config['stages']) == args.n_partitions + 1:
            stages_on_same_gpu = [{0, args.n_partitions}]

        _, summary = run_analysis(analysis_sample,
                                  graph,
                                  analysis_config,
                                  args.n_iter,
                                  recomputation=recomputation,
                                  bw_GBps=bw,
                                  async_pipeline=args.async_pipeline,
                                  sequential_model=model,
                                  stages_on_same_gpu=stages_on_same_gpu)
        with open(f"{args.output_file}.py", "a") as f:
            f.write("\n")
            f.write('"""analysis summary\n' + summary + "\n" + '"""')


def auto_file_name(args):
    s = []
    s.append(args.output_file)
    model_name = args.model_name_or_path.replace("-", "_")
    s.append(model_name)
    s.append(f"p{args.n_partitions}")
    if args.lmhead:
        s.append("lm")
        tied = "tied" if args.stateless_tied else "untied"
        s.append(tied)

    if args.bwd_to_fwd_ratio > 0:
        bwd_to_fwd_ratio = args.bwd_to_fwd_ratio
        if (int(bwd_to_fwd_ratio)) == bwd_to_fwd_ratio:
            bwd_to_fwd_ratio = int(bwd_to_fwd_ratio)
        bwd_to_fwd_ratio = str(bwd_to_fwd_ratio).replace(".", "d")
        s.append(f"r{bwd_to_fwd_ratio}")
    s = "_".join(s)
    return s


class ParsePartitioningOptsLM(ParsePartitioningOpts):
    def _extra(self, parser):
        parser.add_argument("--train_data_file",
                            default=None,
                            type=str,
                            required=True,
                            help="The input training data file (a text file).")
        parser.add_argument("--model_type",
                            default="gpt2",
                            type=str,
                            help="The model architecture to be fine-tuned.")
        parser.add_argument(
            "--model_name_or_path",
            default="gpt2",
            type=str,
            help="The model checkpoint for weights initialization.")
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Optional pretrained config name or path if not the same as model_name_or_path"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path"
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)"
        )
        parser.add_argument(
            "--block_size",
            default=-1,
            type=int,
            help="Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        )
        parser.add_argument(
            "--do_lower_case",
            action='store_true',
            help="Set this flag if you are using an uncased model.")
        parser.add_argument(
            '--overwrite_cache',
            action='store_true',
            help="Overwrite the cached training and evaluation sets")
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")

        parser.add_argument("--lmhead",
                            default=False,
                            action="store_true",
                            help="Partition a model with LM head")

        parser.add_argument(
            "--stateless_tied",
            default=False,
            action="store_true",
            help="Tie weights stateless trick. Note that shared weight may be sent in pipe"
        )

    def set_defaults(self, parser):
        d = {
            # "threads": 20,
            "partitioning_batch_size": 1,
            "n_iter": 10,
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 1,
        }

        parser.set_defaults(**d)


def parse_cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ParsePartitioningOptsLM().add_partitioning_arguments(parser)

    ParseMetisOpts.add_metis_arguments(parser)
    ParseAcyclicPartitionerOpts.add_acyclic_partitioner_arguments(parser)

    args = parser.parse_args()

    if not args.output_file:
        args.output_file = auto_file_name(args)

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    return args


def main():

    args = parse_cli()
    args.METIS_opt = ParseMetisOpts.metis_opts_dict_from_parsed_args(args)
    args.acyclic_opt = ParseAcyclicPartitionerOpts.acyclic_opts_dict_from_parsed_args(args)


    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.model_too_big else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer

    if args.lmhead:
        if args.stateless_tied:
            model_class_dict_to_use = MODEL_CLASSES_LM_HEAD_STATELESS_TIED
        else:
            model_class_dict_to_use = MODEL_CLASSES_LM_HEAD
    else:
        model_class_dict_to_use = MODEL_CLASSES

    config_class, model_class, tokenizer_class = model_class_dict_to_use[
        args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None)

    if not args.save_memory_mode:
        model.to(args.device)
    print("model built")
    train_dataset = load_and_cache_examples(args, tokenizer)
    partition_model(args,
                    train_dataset,
                    model,
                    tokenizer,
                    lm=args.lmhead)


# download dataset from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip

# export TRAIN_FILE=wikitext-2-raw/wiki.train.raw
# python partition_gpt2_models.py --train_data_file=$TRAIN_FILE --no_analysis
# add --dot to get serialized & pdf.
if __name__ == "__main__":
    # For debugging inside docker.
    # import ptvsd
    # port = 1234
    # address = ('0.0.0.0', port)
    # print(f"-I- waiting for attachment on {address}")
    # ptvsd.enable_attach(address=address)
    # ptvsd.wait_for_attach()

    main()

# python partition_gpt2_models.py --use_graph_profiler --profile_ops --analysis_batch_size 1 --async_pipeline --block_size -1 --bwd_to_fwd_ratio 3 --lmhead --model_name_or_path t5-small --train_data_file wikitext-2-raw/wiki.train.raw --model_type t5 --n_iter 50 --n_partitions 2 --output_file results/t5_p2/ --overwrite_cache --partitioning_batch_size 1 --seed 42 --train_data_file wikitext-2-raw/wiki.train.raw
