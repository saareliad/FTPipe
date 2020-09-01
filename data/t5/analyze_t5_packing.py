import numpy as np
import pandas as pd

from data.t5.t5_tfds import like_mtf, get_t5_available_tasks


def density(x):
    return np.count_nonzero(x) / np.prod(x.shape)


def analyze_packing(mixture_or_task_name, sequence_length, dataset_split="train"):
    packed_ds = like_mtf(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                         dataset_split=dataset_split, pack=True)
    ds = packed_ds

    def create_record(packed_example):
        x = packed_example
        # Note: seq len is "max in pack"
        return {
            "input_seq_length": x['inputs_position'].max() + 1,
            "target_seq_length": x['targets_position'].max() + 1,
            "npacked": x['targets_segmentation'].max(),
            "target_density": density(x['targets']),
            "input_density": density(x['inputs']),
        }

    # Note: npacked and density can vary between shuffled iterations.

    df = pd.DataFrame.from_records([create_record(x) for x in ds.as_numpy_iterator()])
    # df.describe()
    return df


def analyze_padding(mixture_or_task_name, sequence_length, dataset_split="train"):
    padded_ds = like_mtf(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                         dataset_split=dataset_split, pack=False)
    ds = padded_ds

    def create_record(padded_example):
        x = padded_example
        return {
            "input_seq_length": np.count_nonzero(x['inputs']),
            "target_seq_length": np.count_nonzero(x['targets']),
            "npacked": 1,
            "target_density": density(x['targets']),
            "input_density": density(x['inputs']),
        }

    df = pd.DataFrame.from_records([create_record(x) for x in ds.as_numpy_iterator()])
    return df


def infer_no_truncation_padding_seq_length(df):
    sequence_length = {
        "inputs": df['input_seq_length'].max(),
        "targets": df['target_seq_length'].max()
    }
    return sequence_length


def infer_no_truncation_padding_seq_length_all_splits(mixture_or_task_name, sequence_length,
                                                      splits=["train", "validation"]):
    df = pd.concat([analyze_packing(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                    dataset_split=dataset_split) for dataset_split in splits])
    return infer_no_truncation_padding_seq_length(df)


def infer_no_truncation_padding_seq_length_for_all_t5_available_tasks():
    names = get_t5_available_tasks(verbose=False)
    # default seq_len
    sequence_length = {
        "inputs": 512,
        "targets": 512
    }
    splits = ["train", "validation"]
    res = {}
    for mixture_or_task_name in names:
        req = infer_no_truncation_padding_seq_length_all_splits(mixture_or_task_name, sequence_length, splits=splits)
        res[mixture_or_task_name] = req

    return res


if __name__ == '__main__':
    mixture_or_task_name = "glue_rte_v002"
    sequence_length = {
        "inputs": 512,
        "targets": 87
    }
    dataset_split = "train"

    df_packing = analyze_packing(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                 dataset_split=dataset_split)

    df_padding = analyze_padding(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                 dataset_split=dataset_split)
