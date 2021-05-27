import ast
import os
import random
import warnings

# import matplotlib as mpl
# mpl.use("pdf")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipe.experiments.experiments import load_experiment


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
                                    checkpoint_every_x_epochs=1, epochs_in_last_checkpoint=None, time_units="hours",
                                    subkey=None):
    times_list = extract_cumsum_train_times(load_experiment(exp_fn), time_units=time_units)
    checkpoints_dict = extract_values(parse_all_eval_results_dict(checkpoints_eval_fn), subkey=subkey)
    # change dict keys according to checkpoint_every_x_epochs
    if checkpoint_every_x_epochs > 1:
        # epochs_in_last
        gpipe_dict_ = {k * checkpoint_every_x_epochs: v for k, v in list(checkpoints_dict.items())[:-1]}

        if epochs_in_last_checkpoint is None:
            epochs_in_last_checkpoint = len(times_list) % checkpoint_every_x_epochs
            warnings.warn(
                f"plot_epochs_vs_accuracy may be inaccurate point for last epoch, inferring it: epochs_in_last_checkpoint={epochs_in_last_checkpoint}")

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


def analyze_datars(times1, times2, values1, values2, colors=('red', 'navy')):
    from adjustText import adjust_text

    all_ts = []

    all_times = [*times1, *times2]
    all_vals = [*values1, *values2]

    for times, values, color in zip([times1, times2], [values1, values2], colors):
        max = np.max(values)
        min = values[0]
        percs = [0.40, 1]
        percs_nice = [str(int((a*100)))+'%' for a in percs] #["40%", "100%"]

        values = np.asarray(values)
        times = np.asarray(times)
        ids = [np.argmax(values >= (x * (max - min) + min)) for x in percs]


        points = [(times[i], values[i], pn) for i, pn in zip(ids, percs_nice)]
        ts = [plt.text(*a, color=color) for a in points]
        all_ts.extend(ts)

        ax = plt.gca()
        annotations = [child for child in ax.get_children() if
                       isinstance(child, matplotlib.text.Annotation) or isinstance(child, matplotlib.legend.Legend
                                                                                   )]

        adjust_text(ts, x=all_times, y=all_vals, add_objects=annotations,
                    arrowprops=dict(arrowstyle="->", fill=True, color=color, ))


def epoch_speedup_dict(exp_gpipe_fn, exp_stale_fn):
    times_gpipe = extract_cumsum_train_times(load_experiment(exp_gpipe_fn))
    times_stale = extract_cumsum_train_times(load_experiment(exp_stale_fn))

    d = epoch_speedup_dict_from_cumsum_times(times_gpipe, times_stale)
    return d


def epoch_speedup_dict_from_cumsum_times(times_gpipe, times_stale):
    try:
        assert len(times_gpipe) == len(times_stale), str((len(times_gpipe), len(times_stale)))
    except AssertionError as e:
        if len(times_gpipe) - len(times_stale) == 1:
            warnings.warn("allowing 1 difference")
            times_gpipe = times_gpipe[:-1]
        elif len(times_stale) - len(times_gpipe) == 2:
            warnings.warn("allowing 2 difference")
            times_stale = times_stale[:-2]
        else:
            raise e

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


def compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                         virtual_stale_dict, virtual_times_gpipe, virtual_times_stale, skip_gpipe_seq=False):

    if not skip_gpipe_seq:
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


class MultiRC:
    @staticmethod
    def all_speedups_multirc():
        subkey='eval/super_glue_multirc_v102/f1'
        ### SEQ
        seq_gpipe_dict, seq_gpipe_times = Hack.get_multirc_seq_hack_gpipe_times_and_dict(subkey=subkey)
        # seq_stale_dict, seq_stale_times = get_rte_seq_hack_stale_times_and_dict()

        exp_results_dir = "results/t5/super_glue/multirc/"
        # seq_stale_fn = "results/FOR_PAPER/all_results_glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_40_se_1_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_40_se_1_seed_42.txt"
        seq_exp_stale_fn = os.path.join(exp_results_dir, "no_virtual_stages_benchmark_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_8_se_4_seed_42.json")
        # seq_exp_stale_fn = os.path.join("results/t5/glue/rte", "glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_40_se_1_seed_42.json")
        seq_stale_fn = "results/all_results_no_virtual_stages_benchmark_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_8_se_4_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_8_se_4_seed_42.txt"
        seq_stale_dict, seq_stale_times = get_fixed_dict_and_times_single(exp_fn=seq_exp_stale_fn,
                                                                                  checkpoints_eval_fn=seq_stale_fn,
                                                                          subkey=subkey)

        ### MPIPE
        exp_results_dir = "results/t5/super_glue/multirc/"
        exp_stale_fn = os.path.join(exp_results_dir, "new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_8_se_2_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir, "new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_8_se_8_seed_42.json")
        gpipe_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_8_se_8_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_8_se_8_seed_42.txt"
        stale_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_8_se_2_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_8_se_2_seed_42.txt"


        virtual_gpipe_dict, virtual_times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn,
                                                                                  checkpoints_eval_fn=gpipe_fn, subkey=subkey)
        virtual_stale_dict, virtual_times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn,
                                                                                  checkpoints_eval_fn=stale_fn, subkey=subkey)

        compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                             virtual_stale_dict, virtual_times_gpipe, virtual_times_stale)

class WIC:
    @staticmethod
    def all_speedups_wic():

        ### SEQ
        # seq_gpipe_dict, seq_gpipe_times = Hack.get_wic_seq_hack_gpipe_times_and_dict()
        # seq_stale_dict, seq_stale_times = get_rte_seq_hack_stale_times_and_dict()

        checkpoint_every_x_epochs = 100 // (5427 // 128)

        seq_stale_fn="results/all_results_no_virtual_stages_benchmark_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_128_se_4_seed_42_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_128_se_4_seed_42.txt"
        seq_exp_stale_fn = os.path.join("results/t5/super_glue/wic", "no_virtual_stages_benchmark_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_128_se_4_seed_42.json")
        seq_stale_dict, seq_stale_times = get_fixed_dict_and_times_single(exp_fn=seq_exp_stale_fn,
                                                                                  checkpoints_eval_fn=seq_stale_fn, checkpoint_every_x_epochs=checkpoint_every_x_epochs)

        ### MPIPE
        exp_results_dir = "results/t5/super_glue/wic/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "new_args_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_128_se_2_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "new_args_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_128_se_8_seed_42.json")

        gpipe_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_128_se_8_seed_42_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_128_se_8_seed_42.txt"
        stale_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_128_se_2_seed_42_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_128_se_2_seed_42.txt"


        virtual_gpipe_dict, virtual_times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn,
                                                                                  checkpoints_eval_fn=gpipe_fn, checkpoint_every_x_epochs=checkpoint_every_x_epochs)
        virtual_stale_dict, virtual_times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn,
                                                                                  checkpoints_eval_fn=stale_fn, checkpoint_every_x_epochs=checkpoint_every_x_epochs)

        seq_gpipe_dict = None
        seq_gpipe_times = None
        compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                             virtual_stale_dict, virtual_times_gpipe, virtual_times_stale, skip_gpipe_seq=True)


class BoolQ:
    @staticmethod
    def all_speedups_boolq():

        ### SEQ
        seq_gpipe_dict, seq_gpipe_times = Hack.get_boolq_seq_hack_gpipe_times_and_dict()
        # seq_stale_dict, seq_stale_times = get_rte_seq_hack_stale_times_and_dict()

        seq_stale_fn = "results/FOR_PAPER/all_results_new_t5_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_20_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_20_se_10_seed_42.txt"
        seq_exp_stale_fn = os.path.join("results/t5/super_glue/boolq", "new_t5_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_20_se_10_seed_42.json")
        seq_stale_dict, seq_stale_times = get_fixed_dict_and_times_single(exp_fn=seq_exp_stale_fn,
                                                                                  checkpoints_eval_fn=seq_stale_fn)

        ### MPIPE
        exp_results_dir = "results/t5/super_glue/boolq/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_20_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_20_se_10_seed_42.json")

        gpipe_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_20_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_20_se_10_seed_42.txt"
        stale_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_20_se_5_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_20_se_5_seed_42.txt"


        virtual_gpipe_dict, virtual_times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn,
                                                                                  checkpoints_eval_fn=gpipe_fn)
        virtual_stale_dict, virtual_times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn,
                                                                                  checkpoints_eval_fn=stale_fn)

        compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                             virtual_stale_dict, virtual_times_gpipe, virtual_times_stale)


class RTE:
    @staticmethod
    def all_speedups_rte():

        ### SEQ
        seq_gpipe_dict, seq_gpipe_times = Hack.get_rte_seq_hack_gpipe_times_and_dict()
        seq_stale_fn = "results/FOR_PAPER/all_results_glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_40_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_40_se_10_seed_42.txt"
        seq_exp_stale_fn = os.path.join("results/t5/glue/rte", "glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_stale_bs_40_se_10_seed_42.json")
        seq_stale_dict, seq_stale_times = get_fixed_dict_and_times_single(exp_fn=seq_exp_stale_fn,
                                                                                  checkpoints_eval_fn=seq_stale_fn)

        ### MPIPE
        # exp_results_dir = "results/t5/super_glue/rte/"
        # exp_stale_fn = os.path.join(exp_results_dir,
        #                             "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        # exp_gpipe_fn = os.path.join(exp_results_dir,
        #                             "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")

        exp_results_dir = "results_b4_20_5_changes/t5/glue/rte"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_results_dir = "results_new_t5/t5/glue/rte"
        exp_gpipe_fn = os.path.join(exp_results_dir, "rte_virtual_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42.json")
        gpipe_fn = "results_new_t5/all_results_rte_virtual_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42.txt"

        # TODO: gpipe needs to re-run
        #### V2
        # exp_gpipe_fn = os.path.join("results/t5/glue/rte", "new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42.json")
        # gpipe_fn = "results/all_results_new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42.txt"

        stale_fn = "results/FOR_PAPER/all_results_new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_40_se_5_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_40_se_5_seed_42.txt"


        virtual_gpipe_dict, virtual_times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn,
                                                                                  checkpoints_eval_fn=gpipe_fn)
        virtual_stale_dict, virtual_times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn,
                                                                                  checkpoints_eval_fn=stale_fn)

        compute_all_speedups(seq_gpipe_dict, seq_gpipe_times, seq_stale_dict, seq_stale_times, virtual_gpipe_dict,
                             virtual_stale_dict, virtual_times_gpipe, virtual_times_stale)


################
class Hack:
    @staticmethod
    def get_rte_seq_hack_gpipe_times_and_dict():
        # FIXME: add new after exp is done.
        #### load virtual stages for accuracy:
        exp_gpipe_fn = "results_new_t5/t5/glue/rte/glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_40_se_10_seed_42.json"
        gpipe_fn = "results_new_t5/all_results_rte_virtual_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42.txt"
        gpipe_fn = "results/all_results_new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_40_se_10_seed_42.txt"
        # results_new_t5/t5/glue/rte/glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_40_se_10_seed_42.json
        # do it so we have some numbers. anyway its negligible noise not worth another 12 hour run...
        d = {"train_epochs_times": [
            397.09888553619385,
            401.3545401096344,
            400.58319187164307,
            401.8276650905609,
            400.83459973335266,
            400.93730449676514
        ]}

        gpipe_dict, times_gpipe = Hack.extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn)

        return gpipe_dict, times_gpipe

    @staticmethod
    def get_boolq_seq_hack_gpipe_times_and_dict():
        # FIXME: add new after exp is done.
        #### load virtual stages for accuracy:

        # TODO: put here the exp of boolq-gpipe-seq
        exp_results_dir = "results/t5/super_glue/boolq/"
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "new_t5_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_20_se_10_seed_42.json")
        # results/t5/super_glue/boolq/new_t5_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_20_se_10_seed_42.json
        # exp_gpipe_fn = "results_new_t5/t5/glue/rte/glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_40_se_10_seed_42.json"
        gpipe_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_20_se_10_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_20_se_10_seed_42.txt"

        # TODO: copy-pasta here from the exp of boolq-gpipe-seq.
        # results_new_t5/t5/glue/rte/glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_40_se_10_seed_42.json
        # do it so we have some numbers. anyway its negligible noise not worth another 12 hour run...

        d = {
            "train_epochs_times":
            extract_train_epoch_times(load_experiment(exp_gpipe_fn))

        }

        # d = {"train_epochs_times": [
        #     397.09888553619385,
        #     401.3545401096344,
        #     400.58319187164307,
        #     401.8276650905609,
        #     400.83459973335266,
        #     400.93730449676514
        # ]}

        gpipe_dict, times_gpipe = Hack.extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn)

        return gpipe_dict, times_gpipe


    @staticmethod
    def get_wic_seq_hack_gpipe_times_and_dict():
        # FIXME: add new after exp is done.
        #### load virtual stages for accuracy:

        # TODO: put here the exp of wic-gpipe-seq
        # exp_gpipe_fn = os.path.join(exp_results_dir,
        #                             "new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_20_se_10_seed_42.json")

        gpipe_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_128_se_8_seed_42_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_128_se_8_seed_42.txt"
        exp_gpipe_fn = "results/t5/super_glue/wic/no_virtual_stages_benchmark_layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_128_se_8_seed_42.json"

        # TODO: copy-pasta here from the exp of wic-gpipe-seq.
        # results_new_t5/t5/glue/rte/glue_rte_12_epochs_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_40_se_10_seed_42.json
        # do it so we have some numbers. anyway its negligible noise not worth another 12 hour run...
        # d = {"train_epochs_times": [
        #     397.09888553619385,
        #     401.3545401096344,
        #     400.58319187164307,
        #     401.8276650905609,
        #     400.83459973335266,
        #     400.93730449676514
        # ]}
        d = {
            "train_epochs_times":
            extract_train_epoch_times(load_experiment(exp_gpipe_fn))

        }
        checkpoint_every_x_epochs = 500 // (5427 // 128)

        gpipe_dict, times_gpipe = Hack.extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn,checkpoint_every_x_epochs=checkpoint_every_x_epochs)

        return gpipe_dict, times_gpipe



    @staticmethod
    def get_multirc_seq_hack_gpipe_times_and_dict(subkey='eval/super_glue_multirc_v102/f1'
):
        # subkey='eval/super_glue_multirc_v102/exact_match'
        # FIXME: add new after exp is done.
        #### load virtual stages for accuracy:
        gpipe_fn = "results/all_results_new_args_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_8_se_8_seed_42_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_async_squad1_mpipe_t5_tfds_gpipe_bs_8_se_8_seed_42.txt"
        #### put here the exp of multirc-gpipe-seq for times
        exp_gpipe_fn = "results/t5/super_glue/multirc/no_virtual_stages_benchmark_layer_graph_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_pipedream_t5_tfds_gpipe_bs_8_se_8_seed_42.json"
        d = {
            "train_epochs_times":
            extract_train_epoch_times(load_experiment(exp_gpipe_fn))

        }
        gpipe_dict, times_gpipe = Hack.extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn, subkey=subkey)
        return gpipe_dict, times_gpipe

    @staticmethod
    def extrapolate_gpipe_times_and_acc_seq(d, exp_gpipe_fn, gpipe_fn, checkpoint_every_x_epochs=1,
                                            epochs_in_last_checkpoint=None,
                                            subkey=None):
        gpipe_virtual_train_epoch_times = extract_train_epoch_times(load_experiment(exp_gpipe_fn))
        gpipe_dict, times_gpipe = get_fixed_dict_and_times_single(exp_fn=exp_gpipe_fn, checkpoints_eval_fn=gpipe_fn,
                                                                  checkpoint_every_x_epochs=checkpoint_every_x_epochs,
                                                                  epochs_in_last_checkpoint=epochs_in_last_checkpoint,
                                                                  # time_units=time_units
                                                                  subkey=subkey
                                                                  )

        times_gpipe_ = d["train_epochs_times"]  # [x*factor for x in times_gpipe]
        assert len(times_gpipe_) == len(times_gpipe),  (len(times_gpipe_), len(times_gpipe))  # FIXME: it should be the same thing.

        while len(times_gpipe_) < len(gpipe_dict) - 1:
            times_gpipe_.append(random.choice(d["train_epochs_times"]))
        # taking last epoch to be proportional
        factor = np.mean(d["train_epochs_times"]) / np.mean(
            gpipe_virtual_train_epoch_times[:len(d["train_epochs_times"])])
        times_gpipe_.append(factor * times_gpipe[-1])
        times_gpipe = times_to_cumsum_and_units(time_units="hours", times=times_gpipe_)
        return gpipe_dict, times_gpipe


class AnnotationPlotsRTE:
    @staticmethod
    def winning_RTE_seq_gpipe_vs_MIXED_stale():
        set_style()
        gpipe_dict, times_gpipe = Hack.get_rte_seq_hack_gpipe_times_and_dict()

        #### GET Virtual
        exp_results_dir = "results_b4_20_5_changes/t5/glue/rte"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_40_se_5_seed_42.json")
        stale_fn = "results_b4_20_5_changes/all_results_new_args_rte_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_40_se_5_seed_42_layer_graph_t5_3b_tied_lmheads_320_8_8p_bw12_async_squad1_mpipe_t5_tfds_stale_bs_40_se_5_seed_42.txt"

        acc_without_ft = 87.72563176895306  # Note: it is the same for both old and new model
        stale_dict, times_stale = get_fixed_dict_and_times_single(exp_fn=exp_stale_fn, checkpoints_eval_fn=stale_fn)

        higher = times_gpipe
        lower = times_stale
        print("epoch_speedup", epoch_speedup_from_cumsum_times(higher, lower))

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)

        ##### EPOCHS
        AnnotationPlotsRTE.winning_epochs(acc_without_ft, gpipe_dict, stale_dict)

        ###### TTA
        AnnotationPlotsRTE.winning_tta(acc_without_ft, gpipe_dict, stale_dict, times_gpipe, times_stale)

    @staticmethod
    def winning_epochs(acc_without_ft, gpipe_dict, stale_dict, dirname="results/paper_plots/",
                       pdfname='new_Final_Plot_winning_RTE_seq_gpipe_vs_MIXED_stale_EPOCHS.pdf'):
        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot([0] + list(gpipe_dict.keys()), [acc_without_ft] + list(gpipe_dict.values()), marker="^",
                label="GPipe", color="navy")
        ax.plot([0] + list(stale_dict.keys()), [acc_without_ft] + list(stale_dict.values()), marker="o",
                label="FTPipe", color="red")
        # ax.set_title("")
        ax.set_ylim(86, 92)
        ax.set_xlabel(f"Epochs")
        ax.set_ylabel(f"Accuracy")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # fig.set_size_inches(width, height)
        analyze_datars([0] + list(stale_dict.keys()),
                       [0] + list(gpipe_dict.keys()),
                       [acc_without_ft] + list(stale_dict.values()),
                       [acc_without_ft] + list(gpipe_dict.values())
                       )
        ax.legend(frameon=False, loc="best", borderaxespad=0)

        os.makedirs(dirname, exist_ok=True)
        fullpdfname = str(os.path.join(dirname, pdfname))
        plt.savefig(fullpdfname, transparent=False, )
        plt.show()

    @staticmethod
    def winning_tta(acc_without_ft, gpipe_dict, stale_dict, times_gpipe, times_stale, dirname="results/paper_plots/",
                    pdfname='new_Final_Plot_winning_RTE_seq_gpipe_vs_MIXED_stale_TTA.pdf'):
        fix, ax = plt.subplots(figsize=(width, height))
        ax.plot([0] + list(times_gpipe), [acc_without_ft] + list(gpipe_dict.values()), marker="^",
                label="GPipe", color="navy")
        ax.plot([0] + list(times_stale), [acc_without_ft] + list(stale_dict.values()), marker="o",
                label="FTPipe", color="red")
        ax.set_ylim(86, 92)
        ax.set_xlabel(f"Time (Hours)")
        ax.set_ylabel(f"Accuracy")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        analyze_datars([0] + list(times_stale),
                       [0] + list(times_gpipe),
                       [acc_without_ft] + list(stale_dict.values()),
                       [acc_without_ft] + list(gpipe_dict.values())
                       )
        ax.legend(frameon=False, loc="best", borderaxespad=0)
        # fig.set_size_inches(width, height)
        os.makedirs(dirname, exist_ok=True)

        fullpdfname = str(os.path.join(dirname, pdfname))
        plt.savefig(fullpdfname, transparent=False, )
        plt.show()


if __name__ == '__main__':
    # TODO: add acc_without_ft acc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str)
    args = parser.parse_args()

    exps = {
        "winning_RTE_seq_gpipe_vs_MIXED_stale": AnnotationPlotsRTE.winning_RTE_seq_gpipe_vs_MIXED_stale,
    }

    all_results = {
        "all_speedups_rte" : RTE.all_speedups_rte,
        "all_speedups_boolq": BoolQ.all_speedups_boolq,
        "all_speedups_wic": WIC.all_speedups_wic,
        "all_speedups_multirc": MultiRC.all_speedups_multirc,
    }

    allplots = {}
    if args.exp in exps:
        exps[args.exp]()
    elif args.exp in all_results:
        all_results[args.exp]()
    elif args.exp in allplots:
        allplots[args.exp]()
    else:
        raise NotImplementedError(f"exp: {args.exp}, available: {list(exps.keys())}")
