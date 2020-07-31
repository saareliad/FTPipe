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

from models.normal import BertForSequenceClassification
from models.normal.NLP_models.modeling_bert import  get_extended_attention_mask
from models.normal.NLP_models.modeling_roberta import RobertaForSequenceClassification
from partition_scripts_utils import (Parser, record_cmdline,
                                     choose_blocks, bruteforce_main)
from partition_async_pipe import partition_async_pipe
from misc import run_analysis,convert_to_analysis_format
from pytorch_Gpipe import pipe_model,get_weight_functions
from pytorch_Gpipe.model_profiling import register_new_traced_function, register_new_explicit_untraced_function
from pytorch_Gpipe.utils import layerDict, tensorDict

from transformers import (MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                          AutoConfig,
                          AutoTokenizer, glue_tasks_num_labels)

from transformers import GlueDataset, GlueDataTrainingArguments

from collections import defaultdict
from torch.utils.data import TensorDataset

import numpy as np
from transformers import EvalPrediction
from transformers.data.metrics import glue_compute_metrics
from transformers.data.processors.glue import (glue_output_modes)
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



# TODO:    "diagnostic"
TASK_NAME_TO_DATA_DIR = {
    'cola': 'CoLA',
    'mnli': 'MNLI',
    'mnli-mm': 'MNLI',
    'mrpc': 'MPRC',
    'sst-2': 'SST-2',
    'sts-b': 'STS-B',
    'qqp': 'QQP',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'wnli': 'WNLI'
}


def get_dataset(args, tokenizer, cache_name="glue_ds.pt"):
    cache_name += args.model_name_or_path
    if os.path.exists(cache_name) and not args.overwrite_cache:
        print(f"-I- loading dataset from cahce {cache_name}...")
        flag = False
        try:
            ds = torch.load(cache_name)
        except Exception as e:
            print("-I- loading from cache failed, creating new dataset. will not overwrite_cache.")
            flag = True
        if not flag:
            return ds

    print("-I- creating dataset")
    data_dir = args.data_dir
    task_dir = TASK_NAME_TO_DATA_DIR.get(args.task_name)
    data_dir = os.path.join(data_dir, task_dir)

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


def get_sample(args, tokenizer, model,analysis=False):
    train_dataset = get_dataset(args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    # TODO: create a dataloader like they do in transformers...
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.analysis_batch_size if analysis else args.partitioning_batch_size)
    batch = next(iter(train_dataloader))

    if args.precompute_attention_mask:
        attention_mask = get_extended_attention_mask(batch[1],batch[0])
    else:
        attention_mask = batch[1]

    inputs = {
        "input_ids": batch[0],
        "attention_mask": attention_mask,
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

    return inputs


class ParsePartitioningOptsGlue(Parser):
    def _add_model_args(self,group):
        group.add_argument("--task_name",
                    type=str,
                    default="mnli",
                    help="Glue task")
        # Required parameters
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
            help="Path to pre-trained model or shortcut name.",
        )
        group.add_argument(
                "--precompute_attention_mask",
                action="store_true",
                default=False,
                help="wether to compute attention mask inside or outside the model"
            )
        group.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help=
            "The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded.",
        )
        group.add_argument(
            "--do_lower_case",
            action="store_true",
            help="Set this flag if you are using an uncased model.")

        def _add_data_args(self,group):
            group.add_argument(
                "--data_dir",
                default="/home_local/saareliad/data/glue_data/",
                type=str,
                help="The input data dir. Should contain the files for the task.")

            group.add_argument(
                "--cache_dir",
                default="",
                type=str,
                help=
                "Where do you want to store the pre-trained models downloaded from s3",
            )
            group.add_argument(
                "--lang_id",
                default=0,
                type=int,
                help=
                "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
            )
            group.add_argument(
                "--overwrite_cache",
                action="store_true",
                help="Overwrite the cached training and evaluation sets")
            group.add_argument("--seed",
                                type=int,
                                default=42,
                                help="random seed for initialization")

    def _default_values(self):
        d = {
            "partitioning_batch_size": 1,
            "n_iter": 1,
            # "model_name_or_path": 'bert',
            # "model_type": 'bert',
            "n_partitions": 2,
            "bw": 12,
            "analysis_batch_size": 1
        }
        return d


def parse_cli():
    parser = ParsePartitioningOptsGlue(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()


    return args


def main(override_dict={}):
    args = parse_cli()
    if override_dict:
        for i, v in override_dict.items():
            setattr(args, i, v)

    args.model_type = args.model_type.lower()

    if not args.output_file:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str(args.model_name_or_path).replace("-", "_")
        args.output_file = f"{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            args.output_file += "_async"

        args.output_file += f"_{args.task_name}"
        args.output_file += "_glue"
    
    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    config = AutoConfig.from_pretrained(args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    
    setattr(config,"precompute_attention_mask",args.precompute_attention_mask)

    # get correct number of labels.
    config.num_labels = glue_tasks_num_labels.get(args.task_name)

    tokenizer = AutoTokenizer.from_pretrained(
       args.model_name_or_path,
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
    model.train()
    sample = get_sample(args, tokenizer, model,analysis=False)

    if not args.save_memory_mode:
        model = model.to(args.device)
        sample = {k:v.to(args.device) for k,v in sample.items()}

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

    sample = get_sample(args, tokenizer, model,analysis=True)
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)


    if not args.no_analysis:
        depth = args.depth
        blocks = args.basic_blocks
        analysis_config = convert_to_analysis_format(config,
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

    print(f"-I- Done: {args.output_file}.py")

    try:
        out = (analysis_result, args)
    except:
        out = args

    return out


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
    
    override_dicts = []  # list of dicts to override args with...

    # TODO: put all hyper parameters here, a dict for each setting we want to try.
    # d1 = dict(basic_blocks=[])
    # ovverride_dicts.append(d1)
    
    NUM_RUNS = 2
    results = {}
    best = None
    TMP = "/tmp/partitioning_outputs/"

    bruteforce_main(main, override_dicts, NUM_RUNS, TMP)

    #  python partition_glue_models.py --model_type bert --objective stage_time --model_name_or_path bert-base-uncased --n_partitions 2


