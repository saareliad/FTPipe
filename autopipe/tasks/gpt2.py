import math
import operator
import os
import pickle
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import GPT2Config, GPT2Tokenizer

from autopipe.autopipe.model_profiling.tracer import (
    register_new_explicit_untraced_function, register_new_traced_function)
from models.normal.NLP_models.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from models.normal.NLP_models.modeling_gpt2_tied_weights import GPT2LMHeadModel as StatelessGPT2LMHeadModel
from models.normal.NLP_models.modeling_gpt2_tied_weights import GPT2Model as StatelessGPT2Model
from . import register_task, Parser
from .partitioning_task import PartitioningTask


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
    return TextDataset(tokenizer,
                       args,
                       file_path=args.train_data_file,
                       block_size=args.block_size)


class ParsePartitioningOptsLM(Parser):
    def _add_model_args(self, group):
        group.add_argument(
            "--model_name_or_path",
            default="gpt2",
            type=str,
            help="The model checkpoint for weights initialization.")
        group.add_argument("--lmhead",
                           default=False,
                           action="store_true",
                           help="Partition a model with LM head")

        group.add_argument(
            "--stateless_tied",
            default=False,
            action="store_true",
            help="Tie weights stateless trick. Note that shared weight may be sent in pipe"
        )
        group.add_argument(
            "--block_size",
            default=-1,
            type=int,
            help="Optional input sequence length after tokenization."
                 "The training dataset will be truncated in block of this size for training."
                 "Default to the model max input length for single sentence inputs (take into account special tokens)."
        )
        group.add_argument(
            "--do_lower_case",
            action='store_true',
            help="Set this flag if you are using an uncased model.")

    def _add_data_args(self, group):
        group.add_argument("--train_data_file",
                           default=None,
                           type=str,
                           required=True,
                           help="The input training data file (a text file).")
        group.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)"
        )
        group.add_argument(
            '--overwrite_cache',
            action='store_true',
            help="Overwrite the cached training and evaluation sets")

    def _default_values(self):
        d = {
            # "threads": 20,
            "partitioning_batch_size": 1,
            "n_iter": 10,
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 1,
            "force_no_recomputation_scopes": ["stateless_lm_head", "lm_head"]
        }

        return d

    def _auto_file_name(self, args) -> str:
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
        return "_".join(s)


class GPT2Partitioner(PartitioningTask):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path,
                                                       do_lower_case=args.do_lower_case,
                                                       cache_dir=args.cache_dir if args.cache_dir else None)

        # NOTE idealy this will be part of _post_parse
        # but we do not have access to the tokenizer there
        if args.block_size <= 0:
            # Our input block size will be the max possible for the model
            args.block_size = self.tokenizer.max_len_single_sentence
        args.block_size = min(args.block_size, self.tokenizer.max_len_single_sentence)

        self.ds = load_and_cache_examples(args, self.tokenizer)

    @property
    def batch_dim(self) -> int:
        return 0

    def post_partitioning(self, args, graph, analysis_result, summary):
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

    def update_analysis_kwargs(self, args, config, analysis_kwargs: Dict) -> Dict[str, Any]:
        stages_on_same_gpu = set()
        if args.lmhead and args.stateless_tied and len(
                config['stages']) == args.n_partitions + 1:
            stages_on_same_gpu = [{0, args.n_partitions}]
        analysis_kwargs['stages_on_same_gpu'] = stages_on_same_gpu
        return analysis_kwargs

    def register_functions(self):
        register_new_traced_function(math.sqrt, namespace=math)
        register_new_explicit_untraced_function(operator.is_,
                                                namespace=operator)
        register_new_explicit_untraced_function(operator.is_not,
                                                namespace=operator)

    def get_model(self, args) -> torch.nn.Module:
        # Load pretrained model and tokenizer
        if args.lmhead:
            if args.stateless_tied:
                model_class = StatelessGPT2LMHeadModel
            else:
                model_class = GPT2LMHeadModel
        elif args.stateless_tied:
            model_class = StatelessGPT2Model
        else:
            model_class = GPT2Model
        model_config = GPT2Config.from_pretrained(args.model_name_or_path,
                                                  cache_dir=args.cache_dir if args.cache_dir else None)

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=model_config,
            cache_dir=args.cache_dir if args.cache_dir else None).train()

        model.resize_token_embeddings(len(self.tokenizer))

        if args.stateless_tied:
            model.make_stateless_after_loaded_tied_and_resized()

        return model

    def get_input(self, args, analysis=False):
        batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size
        sampler = RandomSampler(self.ds)
        dl = DataLoader(self.ds,
                        sampler=sampler,
                        batch_size=batch_size)

        batch = next(iter(dl))
        if args.lmhead:
            sample = {"input_ids": batch, "labels": batch}
        else:
            sample = {"input_ids": batch}

        return sample


# download dataset from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip

# export TRAIN_FILE=wikitext-2-raw/wiki.train.raw
# python partition_gpt2_models.py --train_data_file=$TRAIN_FILE --no_analysis

register_task("gpt2", ParsePartitioningOptsLM, GPT2Partitioner)
