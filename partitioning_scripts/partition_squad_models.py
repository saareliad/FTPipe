import argparse
import torch
import os
import logging
import importlib
import sys
sys.path.append("../")
import math
import operator
import functools
from torch.utils.data import DataLoader, RandomSampler

from models.normal import BertForQuestionAnswering
from models.normal.NLP_models.modeling_bert import get_extended_attention_mask
from partition_scripts_utils import Parser, record_cmdline, choose_blocks
from partition_async_pipe import partition_async_pipe
from analysis import run_analysis,convert_to_analysis_format
from pytorch_Gpipe import pipe_model,get_weight_functions
from pytorch_Gpipe.model_profiling import register_new_traced_function, register_new_explicit_untraced_function
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

#############################
def load_and_cache_examples(args,
                            tokenizer,
                            evaluate=False,
                            output_examples=False):
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
            raise NotImplementedError()
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

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(
            {
                "features": features,
                "dataset": dataset,
                "examples": examples
            }, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


class ParsePartitioningOptsSquad(Parser):
    def _add_model_args(self,group):
        group.add_argument(
            "--model_type",
            default=None,
            type=str,
            required=True,
            help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
        )
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
            help="wether to compute attention mask inside or outside the model"
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

    def _add_data_args(self,group):
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
            "--predict_file",
            default=None,
            type=str,
            help=
            "The input evaluation file. If a data dir is specified, will look for the file there"
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
        group.add_argument("--seed",
                            type=int,
                            default=42,
                            help="random seed for initialization")

        group.add_argument(
            "--threads",
            type=int,
            default=4,
            help="multiple threads for converting example to features")
        group.add_argument(
            "--lang_id",
            default=0,
            type=int,
            help=
            "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
        )

    def _extra(self, group):
        # NOTE: copy and paste from run_squard script.
        # Required parameters
        group.add_argument("--debug", action="store_true", default=False)
    
    def _default_values(self):
        return {
            # "threads": 20,
            "partitioning_batch_size": 1,
            "n_iter": 1,
            "n_partitions": 2,
            "bw": 12,
            "analysis_batch_size": 1,
        }


def parse_cli():
    parser = ParsePartitioningOptsSquad(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()
    args.model_type = args.model_type.lower()

    if not args.output_file:
        args.output_file = f"{args.model_type}_{args.n_partitions}p"
    
    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    return args


def get_inputs_squad(args, tokenizer, model, analysis=False):
    batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size

    train_dataset = load_and_cache_examples(args,
                                            tokenizer,
                                            evaluate=False,
                                            output_examples=False)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size)
    batch = next(iter(train_dataloader))
    batch = tuple(t.to(args.device) for t in batch)


    if args.precompute_attention_mask:
        attention_mask = get_extended_attention_mask(batch[1],batch[0])
    else:
        attention_mask = batch[1]


    inputs = {
        "input_ids": batch[0],
        "attention_mask": attention_mask,
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

    return inputs


def main():
    args = parse_cli()

    if args.debug:
        import ptvsd
        address = ('127.0.0.1', 3000)
        print(f"-I- rank waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()
        print("attached")

    config = AutoConfig.from_pretrained(args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    setattr(config,"precompute_attention_mask",args.precompute_attention_mask)

    model = BertForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    ).train()
    
    # Partition the model
    GET_PARTITIONS_ON_CPU = True
    register_new_explicit_untraced_function(operator.is_, operator)
    register_new_explicit_untraced_function(operator.is_not, operator)
    register_new_traced_function(math.sqrt, math)

    sample = get_inputs_squad(args, tokenizer, model, analysis=False)

    if not args.save_memory_mode:
        model = model.to(args.device)
        sample = {k:t.to(args.device) for k,t in sample.items()}

    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    batch_dim = 0
    args.basic_blocks = choose_blocks(model, args)
    bw = args.bw

    node_weight_function, edge_weight_function = get_weight_functions(
        args, verbose=True)

    print("-I- partitioning...")
    partial_pipe_model = functools.partial(
        pipe_model,
        model,
        batch_dim,
        kwargs=sample,
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
        graph = partition_async_pipe(args, model, 0, kwargs=sample,
                        node_weight_function=node_weight_function,
                        edge_weight_function=edge_weight_function,)
    else:
        graph = partial_pipe_model()
    if args.dot:
        graph.save_as_pdf(args.output_file, ".")
        graph.serialize(args.output_file)

    record_cmdline(args.output_file)
    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration

    sample = get_inputs_squad(args, tokenizer, model, analysis=True)

    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)


    if not args.no_analysis:
        depth = args.depth
        blocks = args.basic_blocks
        analysis_config =convert_to_analysis_format(config,
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

        analysis_result, summary = run_analysis(
            sample,
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