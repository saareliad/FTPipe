import argparse
import torch
import os
import logging
import importlib
import sys
from torch.utils.data import DataLoader, RandomSampler

from models.normal import BertForQuestionAnswering
from models.normal.NLP_models.modeling_bert import SQUAD_loss
from partition_scripts_utils import ParsePartitioningOpts, ParseMetisOpts, record_cmdline
from heuristics import node_weight_function, edge_weight_function
from misc import run_analysis
from pytorch_Gpipe import PipelineConfig, pipe_model
from pytorch_Gpipe.utils import layerDict, tensorDict

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    # WEIGHTS_NAME,
    # AdamW,
    AutoConfig,
    # AutoModelForQuestionAnswering,
    AutoTokenizer,
    # get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
# from transformers.data.metrics.squad_metrics import (
#     compute_predictions_log_probs,
#     compute_predictions_logits,
#     squad_evaluate,
# )

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

logger = logging.getLogger(__name__)

# FIXME
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in MODEL_CONFIG_CLASSES),
    (),
)


#############################
def load_and_cache_examples(args,
                            tokenizer,
                            evaluate=False,
                            output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
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

        if not args.data_dir and ((evaluate and not args.predict_file) or
                                  (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError(
                    "If not data_dir is specified, tensorflow_datasets needs to be installed."
                )

            if args.version_2_with_negative:
                logger.warn(
                    "tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(
                tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor(
            ) if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(
                    args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(
                    args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(
                {
                    "features": features,
                    "dataset": dataset,
                    "examples": examples
                }, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        #  and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


class ParsePartitioningOptsSquad(ParsePartitioningOpts):
    # TODO:
    def __init__(self):
        super().__init__()

    def _extra(self, parser):
        # NOTE: copy and paste from run_squard script.
        # Required parameters
        parser.add_argument(
            "--model_type",
            default=None,
            type=str,
            required=True,
            help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(ALL_MODELS),
        )
        parser.add_argument(
            "--output_dir",
            default=None,
            type=str,
            required=True,
            help="The output directory where the model checkpoints and predictions will be written.",
        )

        # Other parameters
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            help="The input data dir. Should contain the .json files for the task."
            +
            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        parser.add_argument(
            "--train_file",
            default=None,
            type=str,
            help="The input training file. If a data dir is specified, will look for the file there"
            +
            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        parser.add_argument(
            "--predict_file",
            default=None,
            type=str,
            help="The input evaluation file. If a data dir is specified, will look for the file there"
            +
            "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )

        parser.add_argument(
            "--version_2_with_negative",
            action="store_true",
            help="If true, the SQuAD examples contain some that do not have an answer.",
        )
        parser.add_argument(
            "--null_score_diff_threshold",
            type=float,
            default=0.0,
            help="If null_score - best_non_null is greater than the threshold predict null.",
        )

        parser.add_argument(
            "--max_seq_length",
            default=384,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        parser.add_argument(
            "--doc_stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take between chunks.",
        )
        parser.add_argument(
            "--max_query_length",
            default=64,
            type=int,
            help="The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length.",
        )
        parser.add_argument("--do_train",
                            action="store_true",
                            help="Whether to run training.")
        parser.add_argument("--do_eval",
                            action="store_true",
                            help="Whether to run eval on the dev set.")
        parser.add_argument(
            "--evaluate_during_training",
            action="store_true",
            help="Run evaluation during training at each logging step.")
        parser.add_argument(
            "--do_lower_case",
            action="store_true",
            help="Set this flag if you are using an uncased model.")

        parser.add_argument("--per_gpu_train_batch_size",
                            default=8,
                            type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size",
                            default=8,
                            type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument("--weight_decay",
                            default=0.0,
                            type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon",
                            default=1e-8,
                            type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm",
                            default=1.0,
                            type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs",
                            default=3.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument(
            "--max_steps",
            default=-1,
            type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )
        parser.add_argument("--warmup_steps",
                            default=0,
                            type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        )
        parser.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help="The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another.",
        )
        parser.add_argument(
            "--verbose_logging",
            action="store_true",
            help="If true, all of the warnings related to data processing will be printed. "
            "A number of warnings are expected for a normal SQuAD evaluation.",
        )
        parser.add_argument(
            "--lang_id",
            default=0,
            type=int,
            help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
        )

        parser.add_argument("--logging_steps",
                            type=int,
                            default=500,
                            help="Log every X updates steps.")
        parser.add_argument("--save_steps",
                            type=int,
                            default=500,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument(
            "--eval_all_checkpoints",
            action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
        )
        parser.add_argument("--no_cuda",
                            action="store_true",
                            help="Whether not to use CUDA when available")
        parser.add_argument(
            "--overwrite_output_dir",
            action="store_true",
            help="Overwrite the content of the output directory")
        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets")
        parser.add_argument("--seed",
                            type=int,
                            default=42,
                            help="random seed for initialization")

        parser.add_argument("--local_rank",
                            type=int,
                            default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument(
            "--fp16_opt_level",
            type=str,
            default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",
        )
        parser.add_argument("--server_ip",
                            type=str,
                            default="",
                            help="Can be used for distant debugging.")
        parser.add_argument("--server_port",
                            type=str,
                            default="",
                            help="Can be used for distant debugging.")

        parser.add_argument(
            "--threads",
            type=int,
            default=1,
            help="multiple threads for converting example to features")

    def set_defaults(self, parser):
        d = {
            # "model": 'wrn_16x4',
            # "threads": 20,
            "partitioning_batch_size": 1,
            "n_iter": 1,
            "output_file": 'bert',
            "n_partitions": 2,
            "bw": 12,
            "analysis_batch_size": 1,
        }

        parser.set_defaults(**d)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ParsePartitioningOptsSquad().add_partitioning_arguments(parser)
    ParseMetisOpts.add_metis_arguments(parser)

    args = parser.parse_args()

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    return args


def main():
    args = parse_cli()
    METIS_opt = ParseMetisOpts.metis_opts_dict_from_parsed_args(args)

    args.model_type = args.model_type.lower()

    if args.auto_file_name:
        args.output_file = f"{args.model_type}_p{args.n_partitions}"

    #####
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.model_too_big else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    #####

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = BertForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # TODO: if not args.save_memory_mode:
    model.to(args.device)
    model.train()

    train_dataset = load_and_cache_examples(args,
                                            tokenizer,
                                            evaluate=False,
                                            output_examples=False)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.partitioning_batch_size)
    batch = next(iter(train_dataloader))
    batch = tuple(t.to(args.device) for t in batch)

    # BertModel forward: input_ids attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None
    # BertForQuestionAnswering forward
    # input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
    #            start_positions=None, end_positions=None

    # RuntimeError: Type 'Tuple[Tensor, Tensor, Tensor, None, None, Tensor, Tensor]' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and Tuples of Tensors can be traced (toTraceableStack at ../torch/csrc/jit/pybind_utils.h:306)

    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        # # NOTE: we explicitly add to match to the signatute
        # "position_ids": None,
        # "head_mask": None,
        # "start_positions": batch[3],
        # "end_positions": batch[4],
    }

    signature_order = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        # "position_ids",
        # "head_mask",
        # "start_positions",
        # "end_positions",
    ]

    if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
        raise NotImplementedError()
        del inputs["token_type_ids"]

    if args.model_type in ["xlnet", "xlm"]:
        raise NotImplementedError()
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if args.version_2_with_negative:
            inputs.update({"is_impossible": batch[7]})
        if hasattr(model, "config") and hasattr(model.config, "lang2id"):
            inputs.update({
                "langs": (torch.ones(batch[0].shape, dtype=torch.int64) *
                          args.lang_id).to(args.device)
            })

    sample = tuple(inputs[i] for i in signature_order)

    # Partition the model
    # partition_model(args, train_dataset, model, tokenizer, METIS_opt=METIS_opt)
    GET_PARTITIONS_ON_CPU = True

    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    n_partitions = args.n_partitions
    batch_dim = 0
    bwd_to_fwd_ratio = 2
    print("-I- partitioning...")
    graph = pipe_model(model,
                       batch_dim,
                       sample,
                       depth=args.depth,
                       kwargs=None,
                       nparts=n_partitions,
                       output_file=args.output_file,
                       generate_model_parallel=args.generate_model_parallel,
                       use_layers_only_graph=args.partition_layer_graph,
                       node_weight_function=node_weight_function(bwd_to_fwd_ratio=bwd_to_fwd_ratio),
                       edge_weight_function=edge_weight_function(bw),
                       n_iter=n_iter,
                       recomputation=recomputation,
                       save_memory_mode=args.save_memory_mode,
                       METIS_opt=METIS_opt)

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")
        graph.serialize(args.output_file)

    record_cmdline(args.output_file)
    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration

    if GET_PARTITIONS_ON_CPU:
        sample = tuple(t.to("cpu") for t in sample)
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

    pipe_config = PipelineConfig.fromDict(config)
    pipe_config.toJson(f"{args.output_file}.json")

    if not (args.no_test_run and args.no_analysis):
        depth = pipe_config.depth
        blocks = pipe_config.basic_blocks
        analysis_config = pipe_config._to_old_format(
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

    # # Test # TODO: can do it on GPU...
    # if not args.no_test_run:
    #     _ = run_partitions(sample, analysis_config)

    if not args.no_analysis:
        # sample = create_random_sample(args, analysis=True)
        analysis_result, summary = run_analysis(sample,
                                                graph,
                                                analysis_config,
                                                n_iter,
                                                recomputation=recomputation,
                                                bw_GBps=bw,
                                                verbose=True,
                                                async_pipeline=args.async_pipeline,
                                                sequential_model=model)
        with open(f"{args.output_file}.py", "a") as f:
            f.write("\n")
            f.write('"""analysis summary\n' + summary + "\n" + '"""')


if __name__ == "__main__":
    main()
