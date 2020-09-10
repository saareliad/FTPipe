from pprint import pprint

import numpy as np
import pandas as pd

from data.t5.t5_tfds import like_mtf


def density(x):
    return np.count_nonzero(x) / np.prod(x.shape)


def analyze_packing(mixture_or_task_name, sequence_length, dataset_split="train", packed_ds=None):
    if packed_ds is None:
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
    return df


def analyze_padding(mixture_or_task_name, sequence_length, dataset_split="train", padded_ds=None):
    if padded_ds is None:
        padded_ds = like_mtf(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                             dataset_split=dataset_split, pack=False)
    ds = padded_ds

    def create_record(padded_example):
        x = padded_example
        return {
            "input_seq_length": np.count_nonzero(x['inputs']) + 1,
            "target_seq_length": np.count_nonzero(x['targets']) + 1,
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
    df = pd.concat([analyze_padding(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
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
        # "glue_mnli_v002",  # TODO: commented out becasue weird validation names.
        # "glue_mnli_mismatched_v002",
        # "glue_mnli_matched_v002",
        "glue_qnli_v002",
        "glue_rte_v002",
        "glue_wnli_v002",
        # "glue_ax_v002",  # has just "test" split
        # "cnn_dailymail_v002",  # TODO: The version of the dataset you are trying to use (cnn_dailymail/plain_text/1.0.0) is too old for this version of TFDS so cannot be generated.Either sync to a previous version of TFDS to first prepare the data or use another version of the dataset. Available for `download_and_prepare`: ['3.0.0']
        # TODO: don't know which wmt to take.
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
        # "super_glue_axb_v102", only test
        # "super_glue_axg_v102", only test
        # "dpr_v001_simple",  # TODO: Unknown split "validation". Should be one of ['test', 'train'].
        # super_glue_wsc_v102_simple_train
        # super_glue_wsc_v102_simple_eval
        # glue_wnli_v002_simple_eval
        "squad_v010_allanswers",
        # "trivia_qa_v010" # too long
    ]


#######################################
# individual tasks
#######################################
def glue_rte_v002():
    mixture_or_task_name = "glue_rte_v002"
    # Took from GIN
    sequence_length = {
        "inputs": 512,
        "targets": 87
    }
    dataset_split = "train"

    df_packing = analyze_packing(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                 dataset_split=dataset_split)

    df_padding = analyze_padding(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                 dataset_split=dataset_split)

    return df_packing, df_padding


def sum_task(mixture_or_task_name, dataset_split="train", add_percentiles=True):
    # HACK: npacked is detemined by inputs anyway
    sequence_length = {
        "inputs": 512,
        "targets": 512
    }
    df_packing = analyze_packing(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                 dataset_split=dataset_split)

    df_padding = analyze_padding(mixture_or_task_name=mixture_or_task_name, sequence_length=sequence_length,
                                 dataset_split=dataset_split)

    print(40 * "=")
    print("-I- mixture_or_task_name", mixture_or_task_name)
    print("-I- packing:")
    print(df_packing.describe(percentiles=[0.5, 0.75, 0.9, 0.99]))
    print("-I- padding:")
    described_padding = df_padding.describe(percentiles=[0.5, 0.75, 0.9, 0.99])
    print(described_padding)

    splits = ["train"]
    npacked = df_packing['npacked'].mean()
    ntrain = len(df_padding)
    sequence_length_req = infer_no_truncation_padding_seq_length(df_padding)

    record = {
        "mixture_or_task_name": mixture_or_task_name,
        "max_input": sequence_length_req['inputs'],
        "max_targets": sequence_length_req['targets'],
        "npacked": npacked,
        "examples": ntrain,
    }

    if add_percentiles:
        percs_input = {f"input_seq_length_{i}%": described_padding['input_seq_length'][f'{i}%'] for i in
                       [50, 75, 90, 99]}
        percs_target = {f"target_seq_length_{i}%": described_padding['target_seq_length'][f'{i}%'] for i in
                        [50, 75, 90, 99]}

        record.update(percs_input)
        record.update(percs_target)

    print("-I summary:")
    pprint(record)
    print(40 * "=")

    return record


if __name__ == '__main__':
    # df_packing, df_padding = glue_rte_v002()
    pass
    names = t5_tasks_we_want()
    records = [sum_task(n) for n in names]
    df = pd.DataFrame.from_records(records)
    print(df)
