# Based on hugginface transformers commit-id: 33ef7002e17fe42b276dc6d36c07a3c39b1f09ed
import os
import numpy as np
import types
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset

from transformers.data.datasets.glue import GlueDataset, GlueDataTrainingArguments
from transformers.data.metrics import glue_compute_metrics

from transformers.data.processors.glue import (glue_output_modes,
                                               glue_tasks_num_labels)

from transformers import EvalPrediction
from typing import Callable, Dict

from collections import defaultdict


# This being here is HACK is to allow one model for all glue tasks...
class GlueLoss(torch.nn.Module):
    def __init__(self, num_labels):  # config.num_labels
        super().__init__()
        self.num_labels = num_labels
        if self.num_labels == 1:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        if self.num_labels == 1:
            #  We are doing regression
            loss = self.loss(logits.view(-1), labels.view(-1))
        else:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        return loss


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


def glue_data_dir(DATA_DIR):
    return os.path.join(DATA_DIR, "glue_data")


def make_just_y(ds, mode="train"):
    # NOTE: I made it output example ids in eval for conviniece
    y = [feature.label for feature in ds]
    y = torch.tensor(y)
    return TensorDataset(y)


def make_just_x(ds, mode="train"):
    d = defaultdict(list)
    for feature in ds:
        for key, val in vars(feature).items():
            if key == "label":
                continue
            if val is None:
                continue
            d[key].append(val)
    print(d.keys())
    return TensorDataset(*[torch.tensor(x) for x in d.values()])


def getitem(t):
    if isinstance(t, dict):
        res = {i: getitem(v) for i, v in t.items()}
    else:
        try:
            res = t.item()
        except:
            res = t
    return res


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


def get_just_x_or_y_train_dev_dataset(just, DATA_DIR, **kw):
    """ get x or y datset. """
    # NOTE: called with just=just, DATA_DIR=DATA_DIR, **dataset_keywords
    # data_dir = "/home_local/saareliad/data/glue_data/"
    tokenizer = kw["tokenizer"]
    task_name = kw['task_name']
    max_seq_length = kw['max_seq_length']
    overwrite_cache = kw['overwrite_cache']
    data_dir = os.path.join(DATA_DIR, TASK_NAME_TO_DATA_DIR[task_name])
    args = GlueDataTrainingArguments(task_name=task_name,
                                     data_dir=data_dir,
                                     max_seq_length=max_seq_length,
                                     overwrite_cache=overwrite_cache)

    print("-I- creating datasets...")

    train_ds = GlueDataset(args, tokenizer, mode="train")
    dev_ds = GlueDataset(args, tokenizer, mode="dev")
    if just == 'x':
        just_f = make_just_x
    elif just == 'y':
        just_f = make_just_y
    else:
        raise NotImplementedError()

    train_ds = just_f(train_ds, mode='train')
    dev_ds = just_f(dev_ds, mode='eval')

    print("-I- done creating datasets")

    # TODO: handle mnli double eval
    partial_evaluate = build_compute_metrics_fn(task_name)
    num_labels = glue_tasks_num_labels[task_name]

    def evaluate_glue(self):
        global_step = self.fit_res.num_epochs  # TODO
        print("Evaluating Glue on CPU")
        # NOTE: its very super duper dumb but whatever.
        predictions = torch.cat(self.predictions, dim=0).cpu().numpy()
        label_ids = torch.cat(self.label_ids, dim=0).cpu().numpy()
        self.predictions.clear()
        self.label_ids.clear()
        ep = EvalPrediction(predictions, label_ids)
        result = partial_evaluate(ep)

        try:
            print(result)
        except:
            print("evaluate_glue: failed to print result")

        if not hasattr(self.fit_res, 'glue_results'):
            self.fit_res.glue_results = dict()

        self.fit_res.glue_results[global_step] = getitem(result)

    def set_eval(trainer):
        trainer.loss_fn = GlueLoss(num_labels)
        trainer.statistics.evaluate_glue = types.MethodType(
            evaluate_glue, trainer.statistics)
        trainer.statistics.set_glue_task(task_name)

    # NOTE: (examples, features) are needed for evaluation
    return train_ds, dev_ds, set_eval


if __name__ == "__main__":
    from transformers import AutoTokenizer
    # import ptvsd
    # port = 3000 + 0
    # # args.num_data_workers = 0  # NOTE: it does not work without this.
    # address = ('127.0.0.1', port)
    # print(f"-I- rank {0} waiting for attachment on {address}")
    # ptvsd.enable_attach(address=address)
    # ptvsd.wait_for_attach()

    model_name_or_path = "bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased",
        do_lower_case=False,
        cache_dir=None,
    )
