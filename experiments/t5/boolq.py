import ast

import matplotlib.pyplot as plt
import numpy as np

from experiments.experiments import load_experiment


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


def plot_epochs_vs_accuracy(gpipe_fn, stale_fn, c4_acc=None):
    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))
    fix, ax = plt.subplots()

    if c4_acc is None:
        ax.plot(list(gpipe_dict.keys()), list(gpipe_dict.values()), label="gpipe")
        ax.plot(list(stale_dict.keys()), list(stale_dict.values()), label="ours")
    else:
        ax.plot([0] + list(gpipe_dict.keys()), [c4_acc] + list(gpipe_dict.values()), label="gpipe")
        ax.plot([0] + list(stale_dict.keys()), [c4_acc] + list(stale_dict.values()), label="ours")

    ax.legend()
    ax.set_title("super_glue_boolq_accuracy")

    ax.set_xlabel(f"Epochs")
    ax.set_ylabel(f"Accuracy")

    plt.show()


def extract_times(loaded, time_units="seconds"):
    # loaded = load_experiment(exp_stale)
    time_div_factor = {"seconds": 1, "minutes": 60, "hours": 3600}
    time_div_factor = time_div_factor.get(time_units.lower())
    times = loaded[0]['train_epochs_times']
    times = np.array(times) / time_div_factor
    times = np.cumsum(times)
    return times


def plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="hours", c4_acc=None):
    times_gpipe = extract_times(load_experiment(exp_gpipe_fn), time_units=time_units)
    times_stale = extract_times(load_experiment(exp_stale_fn), time_units=time_units)

    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))
    fix, ax = plt.subplots()

    if c4_acc is None:
        ax.plot(times_gpipe, list(gpipe_dict.values()), label="gpipe")
        ax.plot(times_stale, list(stale_dict.values()), label="ours")
    else:
        ax.plot([0] + times_gpipe, [c4_acc] + list(gpipe_dict.values()), label="gpipe")
        ax.plot([0] + times_stale, [c4_acc] + list(stale_dict.values()), label="ours")

    ax.legend()
    ax.set_title("super_glue_boolq_accuracy")

    ax.set_xlabel(f"Time [{time_units}]")
    ax.set_ylabel(f"Accuracy")

    plt.show()


def epoch_speedup_dict(exp_gpipe_fn, exp_stale_fn):
    times_gpipe = extract_times(load_experiment(exp_gpipe_fn))
    times_stale = extract_times(load_experiment(exp_stale_fn))

    assert len(times_gpipe) == len(times_stale)

    d = dict()
    for i in range(len(times_stale)):
        d[i] = times_gpipe[i] / times_stale[i]

    return d


def epoch_speedup(*args, idx=-1, **kwargs):
    return epoch_speedup_dict(*args, **kwargs)[idx]


if __name__ == '__main__':
    # TODO: add c4 acc
    exp_stale_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.json"
    exp_gpipe_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.json"

    gpipe_fn = "results/all_results_test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.txt"
    stale_fn = "results/all_results_test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.txt"

    plot_epochs_vs_accuracy(gpipe_fn, stale_fn, c4_acc=None)
    plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="Hours", c4_acc=None)
    print("epoch_speedup", epoch_speedup())  # 2.518
