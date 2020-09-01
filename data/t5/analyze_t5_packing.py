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
    # names = get_t5_available_tasks(verbose=False)
    names = t5_tasks_we_want()
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


def t5_tasks_we_want():
    return [
            # c4_v020_unsupervised
            # c4_noclean_v020_unsupervised
            # c4_realnewslike_v020_unsupervised
            # c4_webtextlike_v020_unsupervised
            # c4_v220_span_corruption
            # c4_v220_iid_denoising
            # wikipedia_20190301.en_v003_unsupervised
            "glue_cola_v002",
            "glue_sst2_v002",
            "glue_qqp_v002",
            "glue_mrpc_v002",
            "glue_stsb_v002",
            "glue_mnli_v002",
            # "glue_mnli_mismatched_v002",
            # "glue_mnli_matched_v002",
            "glue_qnli_v002",
            "glue_rte_v002",
            "glue_wnli_v002",
            "glue_ax_v002",
            "cnn_dailymail_v002",
            # wmt14_ende_v003
            # wmt14_enfr_v003
            # wmt16_enro_v003
            # wmt15_enfr_v003
            # wmt19_ende_v003
            # wmt_t2t_ende_v003
            "super_glue_boolq_v102",
            "super_glue_cb_v102",
            "super_glue_copa_v102",
            "super_glue_multirc_v102",
            "super_glue_record_v102",
            "super_glue_rte_v102",
            "super_glue_wic_v102",
            "super_glue_axb_v102",
            "super_glue_axg_v102",
            "dpr_v001_simple",
            # super_glue_wsc_v102_simple_train
            # super_glue_wsc_v102_simple_eval
            # glue_wnli_v002_simple_eval
            "squad_v010_allanswers",
            "trivia_qa_v010"]


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
