import os
from typing import Dict

import torch
import transformers
from torch.utils.data import TensorDataset

from pipe.data.datasets import CommonDatasetHandler, register_dataset
from autopipe.autopipe.cache_utils import compute_and_cache
from .preproc import _shift_right, get_attention_mask, get_inverted_encoder_attention_mask
from .t5_tfds_eval import get_t5_sequence_length_from_args


# try:
#     import t5
#     import tensorflow_datasets as tfds
#     import mesh_tensorflow.transformer.dataset as transformer_dataset
#     import tensorflow.compat.v1 as tf
# except Exception as e:
#     print("please use: pip install t5")
#     raise e


def get_t5_available_tasks(verbose=False):
    import t5
    if verbose:
        for i in t5.data.TaskRegistry.names():
            print(i)

    return t5.data.TaskRegistry.names()


def torch_tensor_dict_from_args(args,
                                config,
                                dataset_split="train",
                                preproc_device="cpu"):
    mixture_or_task_name = args.mixture_or_task_name
    sequence_length = get_t5_sequence_length_from_args(args)
    # preproc_device = getattr(args, "preproc_device")
    preproc_batch_size = getattr(args, "preproc_batch_size", 128)

    ds = like_mtf(mixture_or_task_name=mixture_or_task_name,
                  sequence_length=sequence_length,
                  dataset_split=dataset_split,
                  use_cached=False,
                  pack=False)

    return to_torch_tensor_dict(config,
                                ds,
                                preproc_device=preproc_device,
                                preproc_batch_size=preproc_batch_size)


def rte_tensor_dataset(config,
                       preproc_device="cuda:0",
                       preproc_batch_size=128):
    # sequence_length={"inputs": 512, "targets": 84},
    ds = like_mtf(mixture_or_task_name="glue_rte_v002",
                  sequence_length={
                      "inputs": 128,
                      "targets": 16
                  },
                  dataset_split="train",
                  use_cached=False,
                  pack=False)

    d = to_torch_tensor_dict(config, ds, preproc_batch_size, preproc_device)
    tensors = list(d.values())
    tensor_dataset = torch.utils.data.TensorDataset(*tensors)
    return tensor_dataset


def to_torch_tensor_dict(config, ds, preproc_batch_size, preproc_device):
    batched_ds = tokens_to_batches(ds,
                                   preproc_batch_size,
                                   drop_remainder=False)
    batches = []
    for x in batched_ds:
        res_cuda = our_collate_fn(x, torch.device(preproc_device), config)
        keys = list(res_cuda.keys())
        for i in keys:
            res_cuda[i] = res_cuda[i].cpu()
        batches.append(res_cuda)
    del ds
    del batched_ds
    d = {}
    for k in batches[0].keys():
        stacked = torch.cat([b[k] for b in batches])
        d[k] = stacked
    return d


# @torch.no_grad
def our_collate_fn(batch, device, config):
    for i in ['inputs', 'targets', 'inputs_mask', 'targets_mask']:
        batch[i] = torch.from_numpy(batch[i]).to(device)

    input_ids = batch['inputs']
    lm_labels = batch['targets']
    attention_mask = batch['inputs_mask']
    decoder_attention_mask = batch['targets_mask']

    lm_labels[lm_labels[:, :] == 0] = -100

    decoder_input_ids = _shift_right(config, lm_labels)

    precompute_masks = getattr(config, "precomputed_masks", False)
    if precompute_masks:
        # print("-I- precomputing t5 masks on CPU", end ="...")
        inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(
            input_ids.size(), attention_mask, attention_mask.device)
        attention_mask = get_attention_mask(input_ids.size(),
                                            attention_mask,
                                            attention_mask.device,
                                            is_decoder=False)
        decoder_attention_mask = get_attention_mask(
            decoder_input_ids.size(),
            decoder_attention_mask,
            decoder_attention_mask.device,
            is_decoder=True)
        # print("-I- done")
    else:
        # print("-W- preprocessing will happen inside the model...")
        inverted_encoder_attention_mask = None
        # decoder_attention_mask = None

    d = {}
    d['input_ids'] = input_ids
    d['attention_mask'] = attention_mask
    d['decoder_input_ids'] = decoder_input_ids
    d['decoder_attention_mask'] = decoder_attention_mask
    d['inverted_encoder_attention_mask'] = inverted_encoder_attention_mask

    if transformers.__version__ > ('4.1.1'):  # 3.3.1
        d['labels'] = lm_labels
    else:
        d['lm_labels'] = lm_labels

    to_del = [i for i in d if d[i] is None]
    for i in to_del:
        del d[i]

    return d


def like_mtf(mixture_or_task_name: str,
             sequence_length: Dict[str, int],
             dataset_split="train",
             use_cached=False,
             pack=True):
    import t5
    import tensorflow.compat.v1 as tf
    import mesh_tensorflow.transformer.dataset as transformer_dataset

    # https://github.com/google-research/text-to-text-transfer-transformer/blob/5053a463eac3423284c327bf36e61988189239c1/t5/models/mesh_transformer.py

    mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)
    ds = mixture_or_task.get_dataset(sequence_length,
                                     split=dataset_split,
                                     use_cached=use_cached,
                                     shuffle=False)

    def _map_fn(ex):
        for key in ['inputs', 'targets']:
            tensor = ex[key]
            mask = tf.cast(tf.greater(tensor, 0), tensor.dtype)
            ex[key + "_mask"] = mask
        return ex

    feature_keys = tuple(k for k in mixture_or_task.output_features
                         if k in tf.data.get_output_shapes(ds))

    ds = transformer_dataset.pack_or_pad(ds,
                                         sequence_length,
                                         pack=pack,
                                         feature_keys=feature_keys,
                                         ensure_eos=True)

    ds = ds.map(_map_fn)
    """
    # https://github.com/tensorflow/mesh/blob/6f5e3f10b5fe2bbd613bbe11b18a63d52f2749f4/mesh_tensorflow/transformer/utils.py
            encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
            decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                 None),
            decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                    None),
            encoder_position=mtf_features.get("inputs_position", None),
            decoder_position=mtf_features.get("targets_position", None),
    """
    return ds


def tokens_to_batches(dataset, batch_size, drop_remainder=False):
    import tensorflow_datasets as tfds
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return tfds.as_numpy(dataset)


def get_separated_dataset(just, DATA_DIR, args, **dataset_keywords):
    # import tensorflow_datasets as tfds

    config = dataset_keywords['config']
    preproc_device = dataset_keywords.get("preproc_device", "cpu")

    # Get cache names
    name = f"t5_tfds_{args.mixture_or_task_name}_{args.max_seq_length}_{args.answer_max_seq_length}"
    small_cache = "_".join(just) if isinstance(just, list) else just
    big_cache = "FULL"
    ww = ['train', 'val']

    small_cache_name_train = os.path.join(
        DATA_DIR, f"cache_{ww[0]}.{name}_just_{small_cache}.pt")
    small_cache_name_eval = os.path.join(
        DATA_DIR, f"cache_{ww[1]}.{name}_just_{small_cache}.pt")
    big_cache_name_train = os.path.join(
        DATA_DIR, f"cache_{ww[0]}.{name}_just_{big_cache}.pt")
    big_cache_name_eval = os.path.join(
        DATA_DIR, f"cache_{ww[1]}.{name}_just_{big_cache}.pt")

    if isinstance(just, list):
        subset_of_inputs = set(just)
    else:
        raise NotImplementedError()

    def compute_full_train():
        return torch_tensor_dict_from_args(args,
                                           config,
                                           dataset_split="train",
                                           preproc_device=preproc_device)

    def compute_full_eval():
        return torch_tensor_dict_from_args(args,
                                           config,
                                           dataset_split="validation",
                                           preproc_device=preproc_device)

    def compute_subset_from_full(full_func, full_cache_name):
        d = compute_and_cache(full_func, full_cache_name)
        to_drop = [i for i in d.keys() if i not in subset_of_inputs]
        for i in to_drop:
            del d[i]
        d = [torch.tensor(d[i]) for i in just]
        ds = TensorDataset(*d)
        return ds

    def compute_subset_train():
        return compute_subset_from_full(compute_full_train, big_cache_name_train)

    def compute_subset_eval():
        return compute_subset_from_full(compute_full_eval, big_cache_name_eval)

    train_dataset = compute_and_cache(compute_subset_train,
                                      small_cache_name_train)
    dev_dataset = compute_and_cache(compute_subset_eval, small_cache_name_eval)

    return train_dataset, dev_dataset


class SEP_T5_TFDS_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()
        train_ds, test_ds = get_separated_dataset(**kw)
        self.train_ds = train_ds
        self.test_ds = test_ds

    def get_train_ds(self, **kw):
        return self.train_ds

    def get_test_ds(self, **kw):
        return self.test_ds

    def get_validation_ds(self, **kw):
        NotImplementedError()

    def get_modify_trainer_fn(self):
        pass

    def modify_dataloader_keywords(self, dataloader_keywords):
        return dataloader_keywords


register_dataset("t5_tfds", SEP_T5_TFDS_DatasetHandler)
