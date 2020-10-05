import ast
import os
import random
import warnings

# import matplotlib as mpl
# mpl.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.plot import plot_loss
from experiments.experiments import load_experiment


# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')

# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=False)
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)
# def set_style():
#     pass
#     # plt.style.use(['seaborn-white', 'seaborn-paper'])
#     # matplotlib.rc("font", family="Times New Roman")
#     # plt.rc('font', family='serif', serif='Times')

import seaborn as sns
def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


width = 7
height = width / 1.618

# set_style()

GPIPE_MARKER = "^"
STALE_MARKER = "o"


# TODO: scatter plot to mark epochs.

def parse_all_eval_results_dict(fn):
    with open(fn, "r") as f:
        d = ast.literal_eval(f.read())
    return d


def extract_values(d, subkey=None, verbose=False):
    #     d = {"epochs": keys,
    #         "accuracy": values}
    if subkey is None:
        s = set()
        for v in d.values():
            for x in v.keys():
                s.add(x)
        if len(s) == 1:
            subkey = next(iter(s))
        else:
            raise ValueError("please choose subkey from", s)
    if verbose:
        print(f"inferring subkey as {subkey}")

    keys = [1 + x for x in list(d.keys())]
    values = [d[k][subkey] for k in d]
    return {k: v for k, v in zip(keys, values)}


def plot_epochs_vs_accuracy(*, gpipe_dict=None, stale_dict=None, acc_without_ft=None, title="super_glue_boolq_accuracy",
                            ylabel=f"Accuracy"):
    fix, ax = plt.subplots()

    if acc_without_ft is None:
        ax.plot(list(gpipe_dict.keys()), list(gpipe_dict.values()), marker=GPIPE_MARKER, label="gpipe")
        ax.plot(list(stale_dict.keys()), list(stale_dict.values()), marker=STALE_MARKER, label="ours")
    else:
        ax.plot([0] + list(gpipe_dict.keys()), [acc_without_ft] + list(gpipe_dict.values()), marker=GPIPE_MARKER,
                label="gpipe")
        ax.plot([0] + list(stale_dict.keys()), [acc_without_ft] + list(stale_dict.values()), marker=STALE_MARKER,
                label="ours")

    ax.legend()
    ax.set_title(title)

    ax.set_xlabel(f"Epochs")
    ax.set_ylabel(ylabel)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()


def extract_cumsum_train_times(loaded, time_units="seconds"):
    times = extract_train_epoch_times(loaded)

    times = times_to_cumsum_and_units(time_units, times)
    return times


def extract_train_epoch_times(loaded):
    return loaded[0]['train_epochs_times']


def times_to_cumsum_and_units(time_units, times):
    time_div_factor = {"seconds": 1, "minutes": 60, "hours": 3600}
    time_div_factor = time_div_factor.get(time_units.lower())
    times = np.array(times) / time_div_factor
    times = np.cumsum(times)
    return times


def plot_time_vs_accuracy(*, gpipe_dict=None, stale_dict=None, times_gpipe=None, times_stale=None, time_units="hours",
                          acc_without_ft=None,
                          title="super_glue_boolq_accuracy", ylabel=f"Accuracy"):
    fix, ax = plt.subplots()
    # plt.savefig('Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')

    if acc_without_ft is None:
        ax.plot(times_gpipe, list(gpipe_dict.values()), marker=GPIPE_MARKER, label="gpipe")
        ax.plot(times_stale, list(stale_dict.values()), marker=STALE_MARKER, label="ours")
    else:
        ax.plot([0] + list(times_gpipe), [acc_without_ft] + list(gpipe_dict.values()), marker=GPIPE_MARKER,
                label="gpipe")
        ax.plot([0] + list(times_stale), [acc_without_ft] + list(stale_dict.values()), marker=STALE_MARKER,
                label="ours")

    ax.legend()
    ax.set_title(title)

    ax.set_xlabel(f"Time [{time_units}]")
    ax.set_ylabel(ylabel)

    plt.show()


def get_fixed_dict_and_times_single(exp_fn, checkpoints_eval_fn,
                                    checkpoint_every_x_epochs=1, epochs_in_last_checkpoint=None, time_units="hours"):
    times_list = extract_cumsum_train_times(load_experiment(exp_fn), time_units=time_units)
    checkpoints_dict = extract_values(parse_all_eval_results_dict(checkpoints_eval_fn))
    # change dict keys according to checkpoint_every_x_epochs
    if checkpoint_every_x_epochs > 1:
        # epochs_in_last
        gpipe_dict_ = {k * checkpoint_every_x_epochs: v for k, v in list(checkpoints_dict.items())[:-1]}

        if epochs_in_last_checkpoint is None:
            epochs_in_last_checkpoint = len(times_list) % checkpoint_every_x_epochs
            warnings.warn(
                f"plot_epochs_vs_accuracy may be inaccurate point for last epoch, infering it: epochs_in_last_checkpoint={epochs_in_last_checkpoint}")

        print(f"epochs_in_last_checkpoint={epochs_in_last_checkpoint}")
        k, v = list(checkpoints_dict.items())[-1]
        if epochs_in_last_checkpoint == 0:
            gpipe_dict_[k * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v
        else:
            gpipe_dict_[(k - 1) * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v

        times_gpipe_ = [times_list[i] for i in range(0, len(times_list), checkpoint_every_x_epochs)]
        if len(times_list) % checkpoint_every_x_epochs > 0:
            times_gpipe_.append(times_list[-1])

        times_list = times_gpipe_

        checkpoints_dict = gpipe_dict_
    return checkpoints_dict, times_list


def epoch_speedup_dict(exp_gpipe_fn, exp_stale_fn):
    times_gpipe = extract_cumsum_train_times(load_experiment(exp_gpipe_fn))
    times_stale = extract_cumsum_train_times(load_experiment(exp_stale_fn))

    d = epoch_speedup_dict_from_cumsum_times(times_gpipe, times_stale)
    return d


def epoch_speedup_dict_from_cumsum_times(times_gpipe, times_stale):
    assert len(times_gpipe) == len(times_stale)
    d = dict()
    for i in range(len(times_stale)):
        d[i] = times_gpipe[i] / times_stale[i]
    return d


def epoch_speedup_from_cumsum_times(*args, idx=-1, **kwargs):
    return list(epoch_speedup_dict_from_cumsum_times(*args, **kwargs).values())[idx]


def epoch_speedup(*args, idx=-1, **kwargs):
    return list(epoch_speedup_dict(*args, **kwargs).values())[idx]


def dump_all_raw_data(exp_stale_fn, exp_gpipe_fn, gpipe_fn, stale_fn, acc_without_ft=None):
    """  Prints all raw data used for analysis
        The rest are calculations on this data
    """
    print("-I- dump_all_raw_data")
    print(parse_all_eval_results_dict(gpipe_fn))
    print(parse_all_eval_results_dict(stale_fn))
    print(load_experiment(exp_gpipe_fn)[0]['train_epochs_times'])
    print(load_experiment(exp_stale_fn)[0]['train_epochs_times'])
    if acc_without_ft is not None:
        print("result_without_fine_tuning:", acc_without_ft)


def time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale, slow_alg_name='gpipe', fast_alg_name='stale'):
    values_gpipe = list(gpipe_dict.values())
    values_stale = list(stale_dict.values())

    # now do the job

    max_gpipe = np.max(values_gpipe)
    max_stale = np.max(values_stale)

    argmax_gpipe = np.argmax(values_gpipe)
    argmax_stale = np.argmax(values_stale)

    time_to_best_gpipe = times_gpipe[argmax_gpipe]
    time_to_best_stale = times_stale[argmax_stale]

    records = []
    records.append({"alg": slow_alg_name,
                    "best_result": max_gpipe,
                    "best_result_epoch": list(gpipe_dict.keys())[int(argmax_gpipe)],
                    "time": time_to_best_gpipe})
    records.append({"alg": fast_alg_name,
                    "best_result": max_stale,
                    "best_result_epoch": list(stale_dict.keys())[int(argmax_stale)],
                    "time": time_to_best_stale})

    df = pd.DataFrame.from_records(records)
    print(df)
    speedup_to_best = time_to_best_gpipe / time_to_best_stale
    print("speedup_to_best_result:", speedup_to_best)


if __name__ == '__main__':
    # TODO: add acc_without_ft acc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str)
    args = parser.parse_args()


    def boolq_virtual():
        d = {"train_epochs_times": [
            2790.405769586563,
            2794.8142573833466,
            2795.023707151413,
            2792.52312541008,
            2791.8755214214325,
            2789.138890981674,
            # 2175.9098143577576
        ], }
        mean_5_mb = np.mean(d['train_epochs_times'])

        ### GPipe 10 micro batches:
        d = {"train_epochs_times": [
            2161.4724683761597,
            2122.029277086258
        ], }

        mean_10_mb = np.mean(d['train_epochs_times'])

        factor = mean_10_mb / mean_5_mb
        print(f"-W- For 10 micro batches must multiply gpipe times by: {factor}")

        exp_stale_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.json"
        exp_gpipe_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.json"

        gpipe_fn = "results/FOR_PAPER/T5/boolq/boolq_virtual/all_results_test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/boolq/boolq_virtual/all_results_test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.txt"
        acc_without_ft = 87.61467889908256

        warnings.warn("boolq gpipe was used with wrong number of micro batches, fixing... ")

        gpipe_dict, times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn, checkpoints_eval_fn=gpipe_fn)
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn)

        def mul_times_by_factor(l):
            return list(np.asarray(l) * factor)

        times_gpipe = mul_times_by_factor(times_gpipe)

        print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn) * factor)  # 2.518

        plot_epochs_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, acc_without_ft=acc_without_ft,
                                ylabel="Accuracy",
                                title="super_glue_boolq_accuracy (Mixed Pipeline)")
        plot_time_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, times_gpipe=times_gpipe,
                              times_stale=times_stale, time_units="Hours",
                              acc_without_ft=acc_without_ft,
                              ylabel="Accuracy", title="super_glue_boolq_accuracy (Mixed Pipeline)")

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


    def all_speedups_rte():
        seq_gpipe_dict, seq_gpipe_times = get_rte_seq_hack_gpipe_times_and_dict()
        seq_stale_dict, seq_stale_times = get_rte_seq_hack_stale_times_and_dict()

        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")
        gpipe_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_goipe_bs_40_se_10_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"
        acc_without_ft = 87.72563176895306
        virtual_gpipe_dict, virtual_times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn,
                                                                                  checkpoints_eval_fn=gpipe_fn)
        virtual_stale_dict, virtual_times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn,
                                                                                  checkpoints_eval_fn=stale_fn)

        compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                             virtual_stale_dict, virtual_times_gpipe, virtual_times_stale)


    def all_speedups_boolq():
        raise NotImplementedError()


    def all_speedups_wic():
        acc_without_ft = 72.10031347962382
        virtual_gpipe_dict, virtual_stale_dict, virtual_times_gpipe, virtual_times_stale = get_wic_mixed_gpipe_and_stale_stats()
        raise NotImplementedError()


    def wic_stale_mixed_vs_gpipe_seq_epoch_speedup():
        # # FIXME: I got a bug here
        # warnings.warn("buggy wic_stale_mixed_vs_gpipe_seq_epoch_speedup")
        _, _, _, virtual_times_stale = get_wic_mixed_gpipe_and_stale_stats()

        virtual_exp_gpipe_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.json"
        virtual_gpipe_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.txt"
        checkpoint_every_x_epochs = 500 // (5427 // 128)

        d = {
            "train_epochs_times": [
                198.61418557167053,
                199.19361209869385,
                199.2489116191864,
                197.86673521995544,
                199.3534836769104,
                199.9895725250244,
                200.86272764205933,
                200.04046940803528,
                199.35887932777405,
                199.98969101905823,
                198.66443347930908,
                200.1890172958374
            ],
        }

        virtual_exp_stale_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.json"

        m1 = np.mean(d['train_epochs_times'])
        m2 = np.mean(extract_train_epoch_times(load_experiment(virtual_exp_stale_fn))[:len(d['train_epochs_times'])])

        print("epoch_speedup", m1/m2)
        # return
        #
        #
        # _, seq_gpipe_times = extrapolate_gpipe_times_and_acc_seq(d, virtual_exp_gpipe_fn, virtual_gpipe_fn,
        #                                                          checkpoint_every_x_epochs=checkpoint_every_x_epochs,
        #                                                          epochs_in_last_checkpoint=None)
        # print(seq_gpipe_times)
        # print(virtual_times_stale)
        # print("epoch_speedup", epoch_speedup_from_cumsum_times(seq_gpipe_times, virtual_times_stale))


    def boolq_stale_mixed_vs_gpipe_seq_epoch_speedup():
        d = {"train_epochs_times": [
            3151.963354110718,
            3157.5056524276733
        ], }
        # out of 6 epochs
        # gpipe_seq_mean
        m1 = np.mean(d["train_epochs_times"])

        virtual_exp_stale_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.json"

        m2 = np.mean(extract_train_epoch_times(load_experiment(virtual_exp_stale_fn))[:len(d['train_epochs_times'])])

        print("epoch_speedup", m1/m2)


    def compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                             virtual_stale_dict, virtual_times_gpipe, virtual_times_stale):
        # stale mixed vs gpipe seq
        time_to_best_result(seq_gpipe_dict, virtual_stale_dict, seq_gpipe_times, virtual_times_stale,
                            slow_alg_name="gpipe_seq", fast_alg_name="stale_mixed")
        print("epoch_speedup", epoch_speedup_from_cumsum_times(seq_gpipe_times, virtual_times_stale))
        # gpipe mixed bs gpipe seq
        time_to_best_result(seq_gpipe_dict, virtual_gpipe_dict, seq_gpipe_times, virtual_times_gpipe,
                            slow_alg_name="gpipe_seq", fast_alg_name="gpipe_mixed")
        print("epoch_speedup", epoch_speedup_from_cumsum_times(seq_gpipe_times, virtual_times_gpipe))
        # stale mixed vs gpipe mixed
        time_to_best_result(virtual_gpipe_dict, virtual_stale_dict, virtual_times_gpipe, virtual_times_stale,
                            slow_alg_name="gpipe_mixed", fast_alg_name="stale_mixed")
        print("epoch_speedup", epoch_speedup_from_cumsum_times(virtual_times_gpipe, virtual_times_stale))
        # stale mixed bs stale seq
        time_to_best_result(seq_stale_dict, virtual_stale_dict, seq_stale_times, virtual_times_stale,
                            slow_alg_name="stale_seq", fast_alg_name="stale_mixed")
        print("epoch_speedup", epoch_speedup_from_cumsum_times(seq_stale_times, virtual_times_stale, ))


    def rte_virtual():
        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")
        gpipe_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_goipe_bs_40_se_10_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"
        acc_without_ft = 87.72563176895306
        gpipe_dict, times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn, checkpoints_eval_fn=gpipe_fn)
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn)
        plot_epochs_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, acc_without_ft=acc_without_ft,
                                ylabel="Accuracy",
                                title="glue_rte_accuracy (Mixed Pipeline)")
        plot_time_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, times_gpipe=times_gpipe,
                              times_stale=times_stale, time_units="Hours",
                              acc_without_ft=acc_without_ft, ylabel="Accuracy",
                              title="glue_rte_accuracy (Mixed Pipeline)")
        print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


    def rte_seq_hack():

        records = []
        stale_dict, times_stale = get_rte_seq_hack_stale_times_and_dict()
        gpipe_dict, times_gpipe = get_rte_seq_hack_gpipe_times_and_dict()

        # 59: {'eval/glue_rte_v002/accuracy': 90.97472924187726},
        best_result_epochs = np.argmax(list(stale_dict.values()))
        a = time_to_result = times_stale[best_result_epochs]
        records.append({"alg": "seq_stale",
                        "best_result": 90.97472924187726,
                        "best_result_epoch": best_result_epochs + 1,
                        "time": time_to_result})

        best_result_epochs = np.argmax(list(gpipe_dict.values()))
        b = time_to_result = times_gpipe[best_result_epochs]
        records.append({"alg": "seq_stale",
                        "best_result": 90.97472924187726,
                        "best_result_epoch": best_result_epochs + 1,
                        "time": time_to_result})

        df = pd.DataFrame.from_records(records)
        print(df)
        speedup_to_best = b / a
        print("speedup_to_best_result:", speedup_to_best)


    def rte_seq_hack_old():
        d = {"train_epochs_times": [
            314.30374097824097,
            315.83524799346924,
            315.50665950775146,
            315.6022758483887,
            315.73971939086914,
            315.414577960968,
            315.68518114089966,
            316.09878849983215,
            316.4218945503235,
            316.18961668014526,
            315.9019775390625,
            315.962073802948
        ], }
        mean_epoch_time = np.mean(d["train_epochs_times"]) / 3600  # hours
        # starting from 0
        # 59: {'eval/glue_rte_v002/accuracy': 90.97472924187726},

        best_result_epochs = 60
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_stale = time_to_result
        records = []
        records.append({"alg": "seq_stale",
                        "best_result": 90.97472924187726,
                        "best_result_epoch": best_result_epochs,
                        "time": time_to_result})

        ### For Gpipe: accuracy does not change.
        # Only epoch time changes.

        d = {"train_epochs_times": [
            487.2022340297699,
            488.64417576789856,
            487.7520182132721,
            491.34442591667175,
            488.5484371185303,
            488.97134494781494,
            490.6395933628082,
            487.4429671764374,
            488.1168282032013,
            488.0837802886963,
            487.9543607234955,
            486.35694670677185
        ]}

        mean_epoch_time = np.mean(d["train_epochs_times"]) / 3600  # hours
        best_result_epochs = 52  # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_gpipe = time_to_result
        records.append({"alg": "seq_gpipe",
                        "best_result": 90.97472924187726,
                        "best_result_epoch": 52,
                        "time": time_to_best_gpipe})

        df = pd.DataFrame.from_records(records)
        print(df)
        speedup_to_best = time_to_best_gpipe / time_to_best_stale
        print("speedup_to_best_result:", speedup_to_best)


    def rte_seq_full():
        # exp_fn = "results/t5/glue/rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_gpipe_bs_32_se_32_seed_42.json"
        exp_stale_fn = "results/t5/glue/rte/rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_stale_bs_40_se_10_seed_42.json"
        exp_gpipe_fn = "results/t5/glue/rte/rte_momentum_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_gpipe_bs_40_se_10_seed_42.json"
        # Note stale with with lower micro batch!

        gpipe_fn = "results/FOR_PAPER/T5/rte/rte_seq/all_results_rte_gpipe_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_gpipe_bs_40_se_10_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/rte/rte_seq/all_results_rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_stale_bs_40_se_10_seed_42.txt"

        acc_without_ft = 87.72563176895306

        gpipe_dict, times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn, checkpoints_eval_fn=gpipe_fn)
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn)

        plot_epochs_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, acc_without_ft=acc_without_ft,
                                ylabel="Accuracy",
                                title="glue_rte_accuracy (seq)")
        plot_time_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, times_gpipe=times_gpipe,
                              times_stale=times_stale, time_units="Hours",
                              acc_without_ft=acc_without_ft, ylabel="Accuracy", title="glue_rte_accuracy (seq)")

        # print("epoch_speedup", epoch_speedup(exp_fn, exp_stale_fn))

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


    def wic_virtual():
        # Note stale with with lower micro batch!
        acc_without_ft = 72.10031347962382

        gpipe_dict, stale_dict, times_gpipe, times_stale = get_wic_mixed_gpipe_and_stale_stats()

        plot_epochs_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, acc_without_ft=acc_without_ft,
                                ylabel="Accuracy",
                                title="super_glue_wic_accuracy (Mixed Pipeline)")
        plot_time_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, times_gpipe=times_gpipe,
                              times_stale=times_stale, time_units="Hours",
                              acc_without_ft=acc_without_ft, ylabel="Accuracy",
                              title="super_glue_wic_accuracy (Mixed Pipeline)")

        print("epoch_speedup", epoch_speedup_from_cumsum_times(times_gpipe, times_stale))

        # print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


    def get_wic_mixed_gpipe_and_stale_stats():
        exp_gpipe_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.json"
        exp_stale_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.json"
        gpipe_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_wic_stale_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.txt"
        checkpoint_every_x_epochs = 500 // (5427 // 128)
        time_units = "hours"
        epochs_in_last_checkpoint = None
        gpipe_dict, times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn, checkpoints_eval_fn=gpipe_fn,
                                                                  checkpoint_every_x_epochs=checkpoint_every_x_epochs,
                                                                  epochs_in_last_checkpoint=epochs_in_last_checkpoint,
                                                                  time_units=time_units)
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn,
                                                                  checkpoint_every_x_epochs=checkpoint_every_x_epochs,
                                                                  epochs_in_last_checkpoint=epochs_in_last_checkpoint,
                                                                  time_units=time_units)
        return gpipe_dict, stale_dict, times_gpipe, times_stale


    def wic_seq():

        # results / t5 / super_glue / wic / no_virtual_stages_benchmark_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic_t5_tfds_stale_bs_128_se_4_seed_42.json
        # results/t5/super_glue/wic/no_virtual_stages_benchmark_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic_t5_tfds_gpipe_bs_128_se_8_seed_42.json

        records = []

        mean_epoch_time = np.mean(
            [111.74286079406738,
             111.25073051452637,
             111.8271758556366,
             111.4955780506134,
             111.23295140266418,
             111.18024349212646,
             111.15999627113342,
             111.83649349212646,
             111.25810074806213,
             111.22783064842224,
             # 127.64369297027588
             ]) / 3600  # hours

        best_result_epochs = 0  # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_stale = time_to_result
        records.append({"alg": "seq_stale",
                        "best_result": "not ready",
                        "best_result_epoch": best_result_epochs,
                        "time": time_to_best_stale})

        # results / t5 / super_glue / wic / no_virtual_stages_benchmark_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic_t5_tfds_stale_bs_128_se_4_seed_42.json

        """
        # so far Got after 20 checkpoints (starting from 0 ) with, and saving every 2 epochs.
        partial result: 12 {'eval/super_glue_wic_v102/accuracy': 73.04075235109718}
        """

        ####
        ### For Gpipe: accuracy does not change.
        # Only epoch time changes.

        d = {
            "train_epochs_times": [
                198.61418557167053,
                199.19361209869385,
                199.2489116191864,
                197.86673521995544,
                199.3534836769104,
                199.9895725250244,
                200.86272764205933,
                200.04046940803528,
                199.35887932777405,
                199.98969101905823,
                198.66443347930908,
                200.1890172958374
            ],
        }

        # TODO: avoid code copy and extrapolate with 1 function
        # gpipe_dict, stale_dict, times_gpipe, times_stale = get_wic_mixed_gpipe_and_stale_stats()
        # gpipe_dict, times_gpipe = extrapolate_gpipe_times_and_acc_seq(d, virtual_exp_gpipe_fn, virtual_gpipe_fn)

        mean_epoch_time = np.mean(d["train_epochs_times"]) / 3600  # hours
        best_result_epochs = 110  # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_gpipe = time_to_result
        records.append({"alg": "seq_gpipe",
                        "best_result": 74.92163,
                        "best_result_epoch": best_result_epochs,
                        "time": time_to_best_gpipe})

        df = pd.DataFrame.from_records(records)
        print(df)

        # TODO: seq
        # speedup_to_best = time_to_best_gpipe / time_to_best_stale
        # print("speedup_to_best_result:", speedup_to_best)


    def get_rte_seq_hack_gpipe_times_and_dict():

        #### load virtual stages for accuracy:
        exp_results_dir = "results/t5/glue/rte/"
        # exp_stale_fn = os.path.join(exp_results_dir,
        #                             "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")
        gpipe_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_goipe_bs_40_se_10_seed_42.txt"

        # do it so we have some numbers. anyway its negligible noise not worth another 12 hour run...
        d = {"train_epochs_times": [
            487.2022340297699,
            488.64417576789856,
            487.7520182132721,
            491.34442591667175,
            488.5484371185303,
            488.97134494781494,
            490.6395933628082,
            487.4429671764374,
            488.1168282032013,
            488.0837802886963,
            487.9543607234955,
            486.35694670677185
        ]}

        gpipe_dict, times_gpipe = extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn)

        return gpipe_dict, times_gpipe


    def extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn, checkpoint_every_x_epochs=1,
                                            epochs_in_last_checkpoint=None):
        gpipe_virtual_train_epoch_times = extract_train_epoch_times(load_experiment(exp_gpipe_fn))
        # stale_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"
        acc_without_ft = 87.72563176895306
        gpipe_dict, times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn, checkpoints_eval_fn=gpipe_fn,
                                                                  checkpoint_every_x_epochs=checkpoint_every_x_epochs,
                                                                  epochs_in_last_checkpoint=epochs_in_last_checkpoint,
                                                                  # time_units=time_units
                                                                  )

        times_gpipe_ = d["train_epochs_times"]  # [x*factor for x in times_gpipe]
        while len(times_gpipe_) < len(times_gpipe) - 1:
            times_gpipe_.append(random.choice(d["train_epochs_times"]))
        # taking last epoch to be proportional
        factor = np.mean(d["train_epochs_times"]) / np.mean(
            gpipe_virtual_train_epoch_times[:len(d["train_epochs_times"])])
        times_gpipe_.append(factor * times_gpipe[-1])
        times_gpipe = times_to_cumsum_and_units(time_units="hours", times=times_gpipe_)
        return gpipe_dict, times_gpipe


    def get_rte_seq_hack_stale_times_and_dict():
        # since I got acc results but somehow lost epoch times,
        # Infer epoch times based on first 12 epoch since std is super low and negligible in speedup calculations
        d = {"train_epochs_times": [
            314.30374097824097,
            315.83524799346924,
            315.50665950775146,
            315.6022758483887,
            315.73971939086914,
            315.414577960968,
            315.68518114089966,
            316.09878849983215,
            316.4218945503235,
            316.18961668014526,
            315.9019775390625,
            315.962073802948
        ], }
        mean_epoch_time = np.mean(d["train_epochs_times"])  # / 3600  # hours
        #### load virtual stages for epoch times:
        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        ### here we take the REAL accuracy measured on seq since acc was computed and times lost...
        stale_fn = "results/FOR_PAPER/T5/rte/rte_seq/all_results_rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_stale_bs_40_se_10_seed_42.txt"
        # take it only for its length
        stale_fn_to_ignore = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"

        real_dict = extract_values(parse_all_eval_results_dict(stale_fn))
        # take_until = 67
        times_stale_no_cumsum = extract_train_epoch_times(load_experiment(exp_stale_fn))
        factor = mean_epoch_time / np.mean(times_stale_no_cumsum[:len(d["train_epochs_times"])])

        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn,
                                                                  checkpoints_eval_fn=stale_fn_to_ignore)

        times_stale_ = d["train_epochs_times"]  # [x*factor for x in times_gpipe]
        while len(times_stale_) < len(times_stale) - 1:
            times_stale_.append(random.choice(d["train_epochs_times"]))
        # make last proportional...
        times_stale_.append(factor * times_stale[-1])

        times_stale = times_to_cumsum_and_units(time_units="hours", times=times_stale_)
        # now, take our dict...
        it = iter(real_dict.values())
        stale_dict = {i: next(it) for i in stale_dict.keys()}

        return stale_dict, times_stale


    def boolq_seq():
        records = []

        # results/t5/super_glue/boolq/no_virtual_stages_benchmark_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_acyclic_t5_tfds_gpipe_bs_20_se_10_seed_42.json
        ###
        # ASYNC, hack

        d = {"train_epochs_times": [
            1832.0645875930786,
            1847.3265137672424,
            1834.4660325050354,
            1836.2592487335205,
            1833.8545711040497,
            1836.267201423645,
            # 1434.308673620224
        ], }
        #
        # {0: {'eval/super_glue_boolq_v102/accuracy': 88.2262996941896},
        #  1: {'eval/super_glue_boolq_v102/accuracy': 88.34862385321101},
        #  2: {'eval/super_glue_boolq_v102/accuracy': 88.74617737003058},
        #  3: {'eval/super_glue_boolq_v102/accuracy': 88.62385321100918},
        #  4: {'eval/super_glue_boolq_v102/accuracy': 88.92966360856269},
        #  5: {'eval/super_glue_boolq_v102/accuracy': 89.0519877675841},
        #  6: {'eval/super_glue_boolq_v102/accuracy': 89.26605504587157}}
        warnings.warn("could train more here, best was  89.26605504587157")
        mean_epoch_time = np.mean(d["train_epochs_times"]) / 3600  # hours
        best_result_epochs = 6  # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_stale = time_to_result
        records.append({"alg": "seq_stale",
                        "best_result": 89.051988,
                        "best_result_epoch": best_result_epochs,
                        "time": time_to_best_stale})

        # ### GPipe
        d = {"train_epochs_times": [
            3151.963354110718,
            3157.5056524276733
        ], }

        mean_epoch_time = np.mean(d["train_epochs_times"]) / 3600  # hours
        best_result_epochs = 4  # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_gpipe = time_to_result
        records.append({"alg": "seq_gpipe",
                        "best_result": 89.051988,
                        "best_result_epoch": best_result_epochs,
                        "time": time_to_best_gpipe})

        df = pd.DataFrame.from_records(records)
        print(df)

        speedup_to_best = time_to_best_gpipe / time_to_best_stale
        print("speedup_to_best_result:", speedup_to_best)


    def one_loss_plot(fn, legend, fig, step_every):
        config, fit_res = load_experiment(fn)
        loss_per_batch = "loss_per_batch" in config['statistics']
        fig, ax = plot_loss(fit_res, fig=fig, log_loss=False,
                            legend=legend, loss_per_batch=loss_per_batch, step_every=step_every,
                            original_step_every=config['step_every'])

        return fig, ax


    def boolq_loss_plots():
        # boolq
        exp_stale_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.json"
        exp_gpipe_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.json"

        fig, ax = one_loss_plot(fn=exp_stale_fn, legend="stale", fig=None, step_every=10 * 10)
        fig, ax = one_loss_plot(fn=exp_gpipe_fn, legend="gpipe", fig=fig, step_every=10 * 10)

        plt.show()


    def wic_loss_plots():
        # wic
        exp_gpipe_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.json"
        exp_stale_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.json"

        fig, ax = one_loss_plot(fn=exp_stale_fn, legend="stale", fig=None, step_every=2 * 10)
        fig, ax = one_loss_plot(fn=exp_gpipe_fn, legend="gpipe", fig=fig, step_every=8 * 10)

        plt.show()


    def rte_loss_plots():

        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")

        fig, ax = one_loss_plot(fn=exp_stale_fn, legend="stale", fig=None, step_every=5 * 10)
        fig, ax = one_loss_plot(fn=exp_gpipe_fn, legend="gpipe", fig=fig, step_every=10 * 10)

        plt.show()


    def winning_RTE_seq_gpipe_vs_MIXED_stale():
        gpipe_dict, times_gpipe = get_rte_seq_hack_gpipe_times_and_dict()

        #### GET Virtual
        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        stale_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"
        acc_without_ft = 87.72563176895306
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn)
        # plot_epochs_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, acc_without_ft=acc_without_ft, ylabel="Accuracy",
        #                         title="glue_rte_accuracy (virtual stages)")
        # plot_time_vs_accuracy(gpipe_dict=gpipe_dict, stale_dict=stale_dict, times_gpipe=times_gpipe, times_stale=times_stale, time_units="Hours",
        #                       acc_without_ft=acc_without_ft, ylabel="Accuracy",
        #                       title="glue_rte_accuracy (virtual stages)")

        higher = times_gpipe
        lower = times_stale
        print("epoch_speedup", epoch_speedup_from_cumsum_times(higher, lower))

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)

        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot([0] + list(gpipe_dict.keys()), [acc_without_ft] + list(gpipe_dict.values()), marker="^",
                label="GPipe", color="navy")
        ax.plot([0] + list(stale_dict.keys()), [acc_without_ft] + list(stale_dict.values()), marker="o",
                label="FTPipe", color="red")
        ax.legend(frameon=False)
        # ax.set_title("")
        ax.set_ylim(86, 92)
        ax.set_xlabel(f"Epochs")
        ax.set_ylabel(f"Accuracy")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # fig.set_size_inches(width, height)
        os.makedirs("results/paper_plots/", exist_ok=True)
        # bbox_inches = 'tight'
        plt.savefig('results/paper_plots/Final_Plot_winning_RTE_seq_gpipe_vs_MIXED_stale_EPOCHS.pdf', transparent=False)
        plt.show()

        fix, ax = plt.subplots(figsize=(width, height))
        ax.plot([0] + list(times_gpipe), [acc_without_ft] + list(gpipe_dict.values()), marker="^",
                label="GPipe", color="navy")
        ax.plot([0] + list(times_stale), [acc_without_ft] + list(stale_dict.values()), marker="o",
                label="FTPipe", color="red")
        ax.set_ylim(86, 92)
        ax.set_xlabel(f"Time (Hours)")
        ax.set_ylabel(f"Accuracy")
        ax.legend(frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # fig.set_size_inches(width, height)
        os.makedirs("results/paper_plots/", exist_ok=True)
        plt.savefig('results/paper_plots/Final_Plot_winning_RTE_seq_gpipe_vs_MIXED_stale_TTA.pdf', transparent=False, )
        plt.show()


    def virtual_stages_SEQ_us_vs_MIXED_US_stale():
        set_style()

        seq_dict, times_seq = get_rte_seq_hack_stale_times_and_dict()

        #### GET Virtual
        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        stale_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"
        acc_without_ft = 87.72563176895306
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn)

        higher = times_seq
        lower = times_stale
        print("epoch_speedup", epoch_speedup_from_cumsum_times(higher, lower))

        time_to_best_result(seq_dict, stale_dict, times_seq, times_stale, slow_alg_name="stale_seq",
                            fast_alg_name="stale_virtual")

        fix, ax = plt.subplots(figsize=(width, height))
        ax.plot([0] + list(times_seq), [acc_without_ft] + list(seq_dict.values()), marker="^",
                label="Seq-pipe", color="mediumseagreen")
        ax.plot([0] + list(times_stale), [acc_without_ft] + list(stale_dict.values()), marker="o",
                label="Mixed-pipe", color="red")
        ax.set_ylim(86, 92)
        ax.set_xlabel(f"Time (Hours)")
        ax.set_ylabel(f"Accuracy")
        ax.legend(frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # fig.set_size_inches(width, height)
        os.makedirs("results/paper_plots/", exist_ok=True)
        plt.savefig('results/paper_plots/virtual_stages_SEQ_us_vs_MIXED_US_stale_TTA.pdf', transparent=False, )
        plt.show()


    exps = {
        "boolq_virtual": boolq_virtual,
        "boolq_seq": boolq_seq,
        "rte_virtual": rte_virtual,
        "rte_seq": rte_seq_hack,
        "wic_virtual": wic_virtual,
        "wic_seq": wic_seq,
        "winning_RTE_seq_gpipe_vs_MIXED_stale": winning_RTE_seq_gpipe_vs_MIXED_stale,
        "virtual_stages_SEQ_us_vs_MIXED_US_stale": virtual_stages_SEQ_us_vs_MIXED_US_stale,
        "all_speedups_rte": all_speedups_rte,
        "wic_stale_mixed_vs_gpipe_seq_epoch_speedup": wic_stale_mixed_vs_gpipe_seq_epoch_speedup,
        "boolq_stale_mixed_vs_gpipe_seq_epoch_speedup": boolq_stale_mixed_vs_gpipe_seq_epoch_speedup
    }

    allplots = {
        "boolq_loss": boolq_loss_plots,
        "wic_loss": wic_loss_plots,
        "rte_loss": rte_loss_plots

    }
    if args.exp in exps:
        exps[args.exp]()
    elif args.exp in allplots:
        allplots[args.exp]()
    else:
        raise NotImplementedError(f"exp: {args.exp}, available: {list(exps.keys())}")
