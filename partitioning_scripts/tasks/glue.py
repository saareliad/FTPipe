import logging
import math
import operator
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (AutoConfig, AutoTokenizer, GlueDataset,
                          GlueDataTrainingArguments, glue_tasks_num_labels)


from models.normal import BertForSequenceClassification
from models.normal.NLP_models.modeling_bert import get_extended_attention_mask
from models.normal.NLP_models.modeling_roberta import RobertaForSequenceClassification
from pytorch_Gpipe.model_profiling import (register_new_explicit_untraced_function,
                                             register_new_traced_function)

from . import register_task
from .task import Parser, Partitioner




logger = logging.getLogger(__name__)


MODEL_TYPES = ['bert','roberta']


def make_just_x(ds):
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
    ds = make_just_x(ds)

    if (not os.path.exists(cache_name)) or args.overwrite_cache:
        print("-I- dataset saved")
        torch.save(ds, cache_name)

    print("-I- DONE creating dataset")
    return ds


def get_sample(args, tokenizer,analysis=False):
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
    }

    if args.model_type == "bert":
        inputs["token_type_ids"] = batch[2]

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
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets")

    def _default_values(self):
        d = {
            "partitioning_batch_size": 1,
            "n_iter": 1,
            "n_partitions": 2,
            "bw": 12,
            "analysis_batch_size": 1
        }
        return d
    
    def _post_parse(self, args,argv):
        args.model_type = args.model_type.lower()
        return super()._post_parse(args,argv)
    
    def _auto_file_name(self, args) -> str:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str(args.model_name_or_path).replace("-", "_")
        output_file = f"{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            output_file += "_async"

        output_file += f"_{args.task_name}"
        output_file += "_glue"

        return output_file


class GluePartitioner(Partitioner):
    def __init__(self,args) -> None:
        super().__init__(args)

        self.tokenizer = AutoTokenizer.from_pretrained(
       args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    @property
    def batch_dim(self) -> int:
        return 0
    
    def get_input(self, args, analysis):
        return get_sample(args, self.tokenizer,analysis=analysis)

    def get_model(self, args) -> torch.nn.Module:
        config = AutoConfig.from_pretrained(args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)

        setattr(config,"precompute_attention_mask",args.precompute_attention_mask)

        # get correct number of labels.
        config.num_labels = glue_tasks_num_labels.get(args.task_name)
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
        ).train()

        return model
    
    def register_functions(self):
        register_new_explicit_untraced_function(operator.is_, operator)
        register_new_explicit_untraced_function(operator.is_not, operator)
        register_new_traced_function(math.sqrt, math)



register_task("glue",ParsePartitioningOptsGlue,GluePartitioner)
