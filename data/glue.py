# Based on hugginface transformers commit-id: 33ef7002e17fe42b276dc6d36c07a3c39b1f09ed
import os
import types
from collections import defaultdict
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from transformers import EvalPrediction
from transformers.data.datasets.glue import GlueDataset, GlueDataTrainingArguments
from transformers.data.metrics import glue_compute_metrics
from transformers.data.processors.glue import (glue_output_modes,
                                               glue_tasks_num_labels)

from .datasets import CommonDatasetHandler, register_dataset


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
            try:
                loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            except Exception as e:
                print(self.num_labels, logits.shape, logits.view(-1, self.num_labels).shape, labels.shape,
                      labels.view(-1).shape)
                raise e

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


def make_just_y(ds, **kw):
    # NOTE: I made it output example ids in eval for conviniece
    y = [feature.label for feature in ds]
    y = torch.tensor(y)
    return TensorDataset(y)


def get_extended_attention_mask(attention_mask,
                                input_ids,
                                dtype=torch.float32):
    """ Extented attention mask, removing the preprocessing from inside to outside, bert"""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    return extended_attention_mask


def make_just_x(ds, **kw):
    # NOTE: it can be done in few lines with nlp packadge...
    # keys = ds[0].keys()
    # d = {k:v for k in keys}
    # del d["label"

    d = defaultdict(list)
    for feature in ds:
        for key, val in vars(feature).items():
            if key == "label":
                continue
            if val is None:
                continue
            d[key].append(val)

    print(d.keys())
    if "attention_mask" in d:
        if kw['precompute_attention_mask']:
            print("-I- precomputing attention mask")
            batch = list(d.values())
            b1 = torch.tensor(batch[1])
            b0 = torch.tensor(batch[0])
            attetion_mask = get_extended_attention_mask(b1, b0)
            d['attention_mask'] = attetion_mask
        # else:
        #     attention_mask = batch[1]

    return TensorDataset(*[torch.tensor(x) for x in d.values()])


MAP_NAMES_TO_FEATURES = {
    'input0': 'input_ids',
    'input1': 'attention_mask',
    'input2': 'token_type_ids'  # bert
}

LAST_PARTITION_EXTRA_LABELS = {
    'label',
}


def make_just_by_ds(ds, just, **kw):
    assert isinstance(just, list)

    A = set(MAP_NAMES_TO_FEATURES[i] for i in just)
    if kw['is_last_partition']:
        A |= LAST_PARTITION_EXTRA_LABELS

    # cache_name = "_".join(sorted(list(A)))

    d = defaultdict(list)
    for feature in ds:
        for key, val in vars(feature).items():
            if val is None:
                continue  # For exmple, token_type_ids are None for roberta.
            if key in A:
                d[key].append(val)
            #     if key in LAST_PARTITION_EXTRA_LABELS:
            #         d[key].append(val)

    print(d.keys())
    if "attention_mask" in d:
        if kw['precompute_attention_mask']:
            print("-I- precomputing attention mask")
            b1 = torch.tensor(d['attention_mask'])
            if 'input_ids' in d:
                b0 = torch.tensor(d['input_ids'])
            else:
                b0 = torch.tensor([feature.input_ids for feature in ds])
            attetion_mask = get_extended_attention_mask(b1, b0)
            d['attention_mask'] = attetion_mask

    ll = []
    for x in d.values():
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        ll.append(x)

    return TensorDataset(*ll)


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
    is_last_partition = kw.get('is_last_partition')
    precompute_attention_mask = kw['precompute_attention_mask']
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
    elif isinstance(just, list):
        just_f = make_just_by_ds

    else:
        raise NotImplementedError()

    train_ds = just_f(train_ds,
                      just=just,
                      precompute_attention_mask=precompute_attention_mask,
                      is_last_partition=is_last_partition)
    dev_ds = just_f(dev_ds,
                    just=just,
                    precompute_attention_mask=precompute_attention_mask,
                    is_last_partition=is_last_partition)

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


class SEP_GLUE_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()
        d = extract_needed_keywords(**kw)
        train_ds, dev_ds, extra = get_just_x_or_y_train_dev_dataset(**d)
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.extra = extra

    def get_train_ds(self, **kw):
        return self.train_ds

    def get_test_ds(self, **kw):
        return self.dev_ds

    def get_validation_ds(self, **kw):
        NotImplementedError()

    def get_modify_trainer_fn(self):
        return self.extra


register_dataset("glue", SEP_GLUE_DatasetHandler)


def extract_needed_keywords(**kw):
    # here for backward compatibility
    args = kw['args']
    dataset_keywords = dict(
        tokenizer=kw['tokenizer'],
        overwrite_cache=getattr(args, 'overwrite_cache', False),
        task_name=getattr(args, 'glue_task_name'),
        max_seq_length=getattr(args, 'max_seq_length', 128),
        precompute_masks=getattr(args, 'precompute_masks', False),
        precompute_attention_mask=getattr(args,
                                          "precompute_attention_mask",
                                          False),
        is_last_partition=args.stage == args.num_stages - 1)

    return dataset_keywords


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
