import argparse
import torch
import os
import logging
import importlib
import sys
import math
import operator
import functools
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from models.normal import BertForSequenceClassification
from models.normal.NLP_models.modeling_bert import GlueLoss
from models.normal.NLP_models.modeling_roberta import RobertaForSequenceClassification
from partition_scripts_utils import (ParsePartitioningOpts,
                                     ParseAcyclicPartitionerOpts,
                                     ParseMetisOpts, record_cmdline,
                                     choose_blocks, run_x_tries_until_no_fail)
from partition_async_pipe import partition_async_pipe
from heuristics import get_weight_functions
from misc import run_analysis, run_partitions
from pytorch_Gpipe import PipelineConfig, pipe_model
from pytorch_Gpipe.model_profiling import register_new_traced_function, register_new_explicit_untraced_function
from pytorch_Gpipe.utils import layerDict, tensorDict
import os

from transformers import (MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                          AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, glue_tasks_num_labels)

from transformers import GlueDataset, GlueDataTrainingArguments

from collections import defaultdict
from torch.utils.data import TensorDataset

import numpy as np
from transformers import EvalPrediction
from transformers.data.metrics import glue_compute_metrics
from transformers.data.processors.glue import (glue_output_modes,
                                               glue_tasks_num_labels)
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# FIXME
MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def make_just_x(ds, mode="train"):
    d = defaultdict(list)
    for feature in ds:  # no reason to go over everything...
        for key, val in vars(feature).items():
            if key == "label":
                continue
            if val is None:
                continue
            d[key].append(val)

    print(d.keys())
    return TensorDataset(*[torch.tensor(x) for x in d.values()])


def make_just_y(ds, mode="train"):
    # NOTE: I made it output example ids in eval for conviniece
    y = [feature.label for feature in ds]
    y = torch.tensor(y)
    return TensorDataset(y)


def get_dataset(args, tokenizer, cache_name="glue_ds.pt"):
    cache_name += args.model_name_or_path
    if os.path.exists(cache_name) and not args.overwrite_cache:
        print(f"-I- loading dataset from cahce {cache_name}")
        ds = torch.load(cache_name)
        return ds

    print("-I- creating dataset")
    data_dir = args.data_dir
    data_dir = os.path.join(data_dir, "MNLI")

    glue_args = GlueDataTrainingArguments(task_name=args.task_name,
                                          data_dir=data_dir,
                                          max_seq_length=args.max_seq_length,
                                          overwrite_cache=args.overwrite_cache)
    ds = GlueDataset(
        glue_args,
        tokenizer,
        mode="train",
    )
    ds = make_just_x(ds, mode="train")

    if (not os.path.exists(cache_name)) or args.overwrite_cache:
        print("-I- dataset saved")
        torch.save(ds, cache_name)

    print("-I- DONE creating dataset")
    return ds


def get_sample(args, tokenizer, model):
    train_dataset = get_dataset(args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    # TODO: create a dataloader like they do in transformers...
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.partitioning_batch_size)
    batch = next(iter(train_dataloader))
    batch = tuple(t.to(args.device) for t in batch)

    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        # "token_type_ids": batch[2],
        # # NOTE: we explicitly add to match to the signatute
        # "position_ids": None,
        # "head_mask": None,
        # "start_positions": batch[3],
        # "end_positions": batch[4],
    }

    signature_order = [
        "input_ids",
        "attention_mask",
        # "token_type_ids",
        # "position_ids",
        # "head_mask",
        # "start_positions",
        # "end_positions",
    ]

    if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
        # del inputs["token_type_ids"]
        pass
    else:
        inputs["token_type_ids"] = batch[2]
        signature_order.append("token_type_ids")

    if args.model_type in ["xlnet", "xlm"]:
        raise NotImplementedError()
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if hasattr(model, "config") and hasattr(model.config, "lang2id"):
            inputs.update({
                "langs": (torch.ones(batch[0].shape, dtype=torch.int64) *
                          args.lang_id).to(args.device)
            })

    sample = tuple(inputs[i] for i in signature_order)
    return sample


class ParsePartitioningOptsGlue(ParsePartitioningOpts):
    # TODO:
    def __init__(self):
        super().__init__()

    def _extra(self, parser):
        # Me adding manually...
        parser.add_argument("--task_name",
                            type=str,
                            default="mnli",
                            help="Glue task")

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
            help="Path to pre-trained model or shortcut name.",
        )

        # Other parameters
        parser.add_argument(
            "--data_dir",
            default="/home_local/saareliad/data/glue_data/",
            type=str,
            help="The input data dir. Should contain the files for the task.")

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
            help=
            "Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help=
            "Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help=
            "The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )

        parser.add_argument(
            "--do_lower_case",
            action="store_true",
            help="Set this flag if you are using an uncased model.")

        parser.add_argument(
            "--lang_id",
            default=0,
            type=int,
            help=
            "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
        )
        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets")
        parser.add_argument("--seed",
                            type=int,
                            default=42,
                            help="random seed for initialization")

        parser.add_argument(
            "--eval_glue",
            action="store_true",
            default=False,
            help="evaluate glue on dev set to check weight loading")

    def set_defaults(self, parser):
        d = {
            "partitioning_batch_size": 1,
            "n_iter": 1,
            # "model_name_or_path": 'bert',
            # "model_type": 'bert',
            "n_partitions": 2,
            "bw": 12,
            "analysis_batch_size": 1,
            'eval_glue': False # FIXME: currently always on until bug is solved
        }

        parser.set_defaults(**d)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ParsePartitioningOptsGlue().add_partitioning_arguments(parser)
    ParseMetisOpts.add_metis_arguments(parser)
    ParseAcyclicPartitionerOpts.add_acyclic_partitioner_arguments(parser)

    args = parser.parse_args()

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    return args


def main():
    args = parse_cli()
    args.METIS_opt = ParseMetisOpts.metis_opts_dict_from_parsed_args(args)
    args.acyclic_opt = ParseAcyclicPartitionerOpts.acyclic_opts_dict_from_parsed_args(
        args)
    args.model_type = args.model_type.lower()

    if not args.output_file:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str(args.model_name_or_path).replace("-", "_")
        args.output_file = f"{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            args.output_file += "_async"

        args.output_file += f"_{args.task_name}"
        args.output_file += "_glue"

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

    # get correct number of labels.
    config.num_labels = glue_tasks_num_labels.get(args.task_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model_cls = {
        'bert': BertForSequenceClassification,
        'roberta': RobertaForSequenceClassification
    }
    model_cls = model_cls[args.model_type]

    model = model_cls.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # TODO: if not args.save_memory_mode:
    model.to(args.device)
    model.train()

    sample = get_sample(args, tokenizer, model)

    # Partition the model
    GET_PARTITIONS_ON_CPU = True
    register_new_explicit_untraced_function(operator.is_, operator)
    register_new_explicit_untraced_function(operator.is_not, operator)
    register_new_traced_function(math.sqrt, math)

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
        args=sample,
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
        graph = partition_async_pipe(args, model, 0, sample,
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

    if GET_PARTITIONS_ON_CPU:
        sample = tuple(t.to("cpu") for t in sample)
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

    pipe_config = PipelineConfig.fromDict(config)

    if not (args.no_test_run and args.no_analysis):
        depth = pipe_config.depth
        blocks = pipe_config.basic_blocks
        analysis_config = pipe_config._to_old_format(
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

    if not args.no_analysis:
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

        if getattr(args, "eval_glue", False):
            n_partitions = sum(1 for k in analysis_config
                               if isinstance(k, int))
            for i in range(n_partitions):
                analysis_config[i]['model'] = analysis_config[i]['model'].eval(
                )
                analysis_config[i]['model'].device = device

            # NOTE: this is the original model
            # With the original model we get sota valid results.
            #otherwise- its random chance
            TAKE_ORIGINAL_MODEL = False

            if TAKE_ORIGINAL_MODEL:
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name_or_path)

            model.to(args.device)
            model.eval()
            data_dir = args.data_dir
            data_dir = os.path.join(data_dir, "MNLI")

            glue_args = GlueDataTrainingArguments(
                task_name=args.task_name,
                data_dir=data_dir,
                max_seq_length=args.max_seq_length,
                overwrite_cache=args.overwrite_cache)

            print("creating datasets")

            CACHE_NAME = "glue_dev" + "_" + args.model_name_or_path
            ds = GlueDataset(
                glue_args,
                tokenizer,
                mode="dev",
            )

            if not os.path.exists(CACHE_NAME + "x.pt"):
                dsx = make_just_x(ds, mode="eval")
                torch.save(dsx, CACHE_NAME + "x.pt")
            else:
                dsx = torch.load(CACHE_NAME + "x.pt")

            if not os.path.exists(CACHE_NAME + "y.pt"):
                dsy = make_just_y(ds, mode="eval")
                torch.save(dsy, CACHE_NAME + "y.pt")
            else:
                dsy = torch.load(CACHE_NAME + "y.pt")

            # TODO: create a dataloader like they do in transformers...
            dlx = DataLoader(dsx,
                             sampler=SequentialSampler(dsx),
                             batch_size=args.analysis_batch_size)

            dly = DataLoader(dsy,
                             sampler=SequentialSampler(dsy),
                             batch_size=args.analysis_batch_size)

            preds = []
            labels = []
            print("running partitions")
            with torch.no_grad():
                for bx, by in zip(dlx, dly):

                    def to_device(a):
                        if isinstance(a, torch.Tensor):
                            return a.to(args.device),
                        else:
                            return [t.to(args.device) for t in a]

                    bx = to_device(bx)
                    # by = to_device(by)

                    logits = model(*bx)

                    if isinstance(logits, tuple):
                        logits = logits[0]

                    # logits = run_partitions(bx, analysis_config)
                    # assert len(logits) == 1
                    # logits = logits[0]

                    preds.append(logits.detach().cpu())
                    labels.append(by[0].detach().cpu())

            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()

            print(preds)
            print(labels)

            ep = EvalPrediction(preds, labels)

            print("evaluating on CPU")
            partial_evaluate = build_compute_metrics_fn(args.task_name)

            result = partial_evaluate(ep)
            print(result)


# def _example():
#     data_dir = "/home_local/saareliad/data/glue_data/"
#     data_dir = os.path.join(data_dir, "MNLI")
#     args = GlueDataTrainingArguments(task_name='mnli',
#                                      data_dir=data_dir,
#                                      max_seq_length=128,
#                                      overwrite_cache=False)
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     ds = GlueDataset(args, tokenizer, mode="train")
#     # NOTE: this is problematic as we have our own implementation
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "bert-base-uncased")


def build_compute_metrics_fn(
        task_name: str) -> Callable[[EvalPrediction], Dict]:

    try:
        # num_labels = glue_tasks_num_labels[task_name]
        output_mode = glue_output_modes[task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (task_name))

    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        import ptvsd
        address = ('127.0.0.1', 3000)
        print(f"-I- rank waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()
        print("attached")

    main()
    #  python partition_glue_models.py --model_type bert --objective stage_time --model_name_or_path bert-base-uncased --n_partitions 2
