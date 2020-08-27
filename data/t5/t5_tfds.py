import torch
from typing import Dict
import os

from .t5_squad import get_inverted_encoder_attention_mask, _shift_right, get_attention_mask, load_huggingface_checkpoint
from data.utils import compute_and_cache

try:
    import t5
    import tensorflow_datasets as tfds
    import mesh_tensorflow.transformer.dataset as transformer_dataset
    import tensorflow.compat.v1 as tf
except Exception as e:
    print("please use: pip install t5")
    raise e

from torch.utils.data import TensorDataset
from data.datasets import CommonDatasetHandler, register_dataset
from .t5_tfds_eval import T5Evaluator
from experiments.experiments import  auto_file_name

def get_t5_available_tasks(verbose=False):
    if verbose:
        for i in t5.data.TaskRegistry.names():
            print(i)

    return t5.data.TaskRegistry.names()


def get_t5_sequence_length_from_args(args):
    return {
        "inputs": args.max_seq_length,
        "targets": args.answer_max_seq_length
    }

def torch_tensor_dict_from_args(args,
                                config,
                                dataset_split=tfds.Split.TRAIN,
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
                  dataset_split=tfds.Split.TRAIN,
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
        decoder_attention_mask = None

    d = {}
    d['input_ids'] = input_ids
    d['attention_mask'] = attention_mask
    d['decoder_input_ids'] = decoder_input_ids
    d['decoder_attention_mask'] = decoder_attention_mask
    d['inverted_encoder_attention_mask'] = inverted_encoder_attention_mask
    d['lm_labels'] = lm_labels

    return d


def like_mtf(mixture_or_task_name: str,
             sequence_length: Dict[str, int],
             dataset_split=tfds.Split.TRAIN,
             use_cached=False,
             pack=True):
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

    ds = ds.map(_map_fn,
                num_parallel_calls=t5.data.preprocessors.num_parallel_calls())
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
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return tfds.as_numpy(dataset)


def get_separated_dataset(just, DATA_DIR, args, **dataset_keywords):

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
                                           dataset_split=tfds.Split.TRAIN,
                                           preproc_device=preproc_device)

    def compute_full_eval():
        return torch_tensor_dict_from_args(args,
                                           config,
                                           dataset_split=tfds.Split.VALIDATION,
                                           preproc_device=preproc_device)

    def compute_subset_train():
        d = compute_and_cache(compute_full_train, big_cache_name_train)
        to_drop = [i for i in d.keys() if i not in subset_of_inputs]
        # train_dataset.drop(to_drop)
        for i in to_drop:
            del d[i]
        d = [torch.tensor(d[i]) for i in just]
        train_dataset = TensorDataset(*d)
        return train_dataset

    def compute_subset_eval():
        d = compute_and_cache(compute_full_eval, big_cache_name_eval)
        to_drop = [i for i in d.keys() if i not in subset_of_inputs]
        # train_dataset.drop(to_drop)
        for i in to_drop:
            del d[i]
        d = [torch.tensor(d[i]) for i in just]
        dev_dataset = TensorDataset(*d)
        return dev_dataset

    train_dataset = compute_and_cache(compute_subset_train,
                                      small_cache_name_train)
    dev_dataset = compute_and_cache(compute_subset_eval, small_cache_name_eval)

    return train_dataset, dev_dataset



#
# def evaluate_tfds_checkpoint(args, cp_number):
#     hugg, tokenizer = load_huggingface_checkpoint(args, cp_number)
#
#     # valid_dataset = compute_and_cache(get_squad_validation_dataset, 'squad_valid_data.pt', args=args,
#     #                                   tokenizer=tokenizer)
#     # # TODO: this can be done smarter, distributed
#     #
#     # batch_size = getattr(args, "single_worker_eval_batch_size", 32)
#     # dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
#     # answers = get_answers(args, hugg, tokenizer, dataloader=dataloader)
#     # valid_dataset.set_format()
#     # squad_result = evaluate_squad_answers(valid_dataset=valid_dataset, answers=answers)
#     # print(squad_result)
#     # return squad_result
#
def evaluate_t5_tfds(args, cp_number, device="cpu"):
    model_dir = auto_file_name(args)
    batch_size = getattr(args, "single_worker_eval_batch_size", 32)
    generate_kwargs = getattr(args, "generate_kwargs", {})
    # generate_kwargs['max_length'] = args.answer_max_length
    evaluator = T5Evaluator(args, model_dir=model_dir, device=device, model=None)
    results = evaluator.eval(mixture_or_task_name=args.mixture_or_task_name,
                   sequence_length=get_t5_sequence_length_from_args(args),
                   batch_size=batch_size, checkpoint_steps=cp_number, split="validation",
                   summary_dir=None,
                   **generate_kwargs
                   )
    return results


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
