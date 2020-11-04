import logging
import math
import operator
import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    BertConfig,
    BertTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

from autopipe.autopipe.model_profiling import register_new_explicit_untraced_function, register_new_traced_function
from models.normal.NLP_models.modeling_bert import BertForQuestionAnswering, get_extended_attention_mask

logger = logging.getLogger(__name__)

from . import register_task, Parser
from .partitioning_task import PartitioningTask


def load_and_cache_examples(args,
                            tokenizer):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (not args.train_file):
            raise NotImplementedError()
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            examples = processor.get_train_examples(
                args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(
            {
                "features": features,
                "dataset": dataset,
                "examples": examples
            }, cached_features_file)

    return dataset


class ParsePartitioningOptsSquad(Parser):
    def _add_model_args(self, group):
        group.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help=
            "Path to pre-trained model or shortcut name in huggingface/models"
        )
        group.add_argument(
            "--precompute_attention_mask",
            action="store_true",
            default=False,
            help="whether to compute attention mask inside or outside the model"
        )
        group.add_argument(
            "--max_seq_length",
            default=384,
            type=int,
            help=
            "The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        group.add_argument(
            "--max_query_length",
            default=64,
            type=int,
            help=
            "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length.",
        )
        group.add_argument(
            "--do_lower_case",
            action="store_true",
            help="Set this flag if you are using an uncased model.")

    def _add_data_args(self, group):
        group.add_argument(
            "--data_dir",
            default=None,
            type=str,
            help=
            "The input data dir. Should contain the .json files for the task."
            +
            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        group.add_argument(
            "--train_file",
            default=None,
            type=str,
            help=
            "The input training file. If a data dir is specified, will look for the file there"
            +
            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        group.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help=
            "Where do you want to store the pre-trained models downloaded from s3",
        )
        group.add_argument(
            "--version_2_with_negative",
            action="store_true",
            help=
            "If true, the SQuAD examples contain some that do not have an answer.",
        )
        group.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help=
            "When splitting up a long document into chunks, how much stride to take between chunks.",
        )
        group.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets")
        group.add_argument(
            "--threads",
            type=int,
            default=4,
            help="multiple threads for converting example to features")

    def _default_values(self):
        return {
            # "threads": 20,
            "partitioning_batch_size": 1,
            "n_iter": 1,
            "n_partitions": 2,
            "bw": 12,
            "analysis_batch_size": 1,
        }

    def _auto_file_name(self, args) -> str:
        return f"bert_{args.n_partitions}p"


class BertPartitioner(PartitioningTask):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                                       do_lower_case=args.do_lower_case,
                                                       cache_dir=args.cache_dir if args.cache_dir else None)

    @property
    def batch_dim(self) -> int:
        return 0

    def get_model(self, args) -> torch.nn.Module:
        config = BertConfig.from_pretrained(args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
        setattr(config, "precompute_attention_mask", args.precompute_attention_mask)

        model = BertForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None).train()

        return model

    def get_input(self, args, analysis):
        return get_inputs_squad(args, self.tokenizer, analysis=analysis)

    def register_functions(self):
        register_new_explicit_untraced_function(operator.is_, operator)
        register_new_explicit_untraced_function(operator.is_not, operator)
        register_new_traced_function(math.sqrt, math)


def get_inputs_squad(args, tokenizer, analysis=False):
    batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size

    train_dataset = load_and_cache_examples(args,
                                            tokenizer)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size)
    batch = next(iter(train_dataloader))
    batch = tuple(t.to(args.device) for t in batch)

    if args.precompute_attention_mask:
        attention_mask = get_extended_attention_mask(batch[1], batch[0])
    else:
        attention_mask = batch[1]

    inputs = {
        "input_ids": batch[0],
        "attention_mask": attention_mask,
        "token_type_ids": batch[2],
    }
    return inputs


register_task("bert_squad", ParsePartitioningOptsSquad, BertPartitioner)
