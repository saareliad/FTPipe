import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=None, title="super_glue_boolq_accuracy",
                            ylabel=f"Accuracy"):
    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))
    fix, ax = plt.subplots()

    if acc_without_ft is None:
        ax.plot(list(gpipe_dict.keys()), list(gpipe_dict.values()), label="gpipe")
        ax.plot(list(stale_dict.keys()), list(stale_dict.values()), label="ours")
    else:
        ax.plot([0] + list(gpipe_dict.keys()), [acc_without_ft] + list(gpipe_dict.values()), label="gpipe")
        ax.plot([0] + list(stale_dict.keys()), [acc_without_ft] + list(stale_dict.values()), label="ours")

    ax.legend()
    ax.set_title(title)

    ax.set_xlabel(f"Epochs")
    ax.set_ylabel(ylabel)

    plt.show()


def extract_times(loaded, time_units="seconds"):
    # loaded = load_experiment(exp_stale)
    time_div_factor = {"seconds": 1, "minutes": 60, "hours": 3600}
    time_div_factor = time_div_factor.get(time_units.lower())
    times = loaded[0]['train_epochs_times']
    times = np.array(times) / time_div_factor
    times = np.cumsum(times)
    return times


def plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="hours", acc_without_ft=None,
                          title="super_glue_boolq_accuracy", ylabel=f"Accuracy"):
    times_gpipe = extract_times(load_experiment(exp_gpipe_fn), time_units=time_units)
    times_stale = extract_times(load_experiment(exp_stale_fn), time_units=time_units)

    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))
    fix, ax = plt.subplots()

    if acc_without_ft is None:
        ax.plot(times_gpipe, list(gpipe_dict.values()), label="gpipe")
        ax.plot(times_stale, list(stale_dict.values()), label="ours")
    else:
        ax.plot([0] + list(times_gpipe), [acc_without_ft] + list(gpipe_dict.values()), label="gpipe")
        ax.plot([0] + list(times_stale), [acc_without_ft] + list(stale_dict.values()), label="ours")

    ax.legend()
    ax.set_title(title)

    ax.set_xlabel(f"Time [{time_units}]")
    ax.set_ylabel(ylabel)

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


def time_to_best_result(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="hours"):
    times_gpipe = extract_times(load_experiment(exp_gpipe_fn), time_units=time_units)
    times_stale = extract_times(load_experiment(exp_stale_fn), time_units=time_units)

    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))

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
    records.append({"alg": "gpipe",
                    "best_result": max_gpipe,
                    "best_result_epoch": argmax_gpipe,
                    "time": time_to_best_gpipe})
    records.append({"alg": "stale",
                    "best_result": max_stale,
                    "best_result_epoch": argmax_stale,
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


    def boolq():
        exp_stale_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.json"
        exp_gpipe_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.json"

        gpipe_fn = "results/FOR_PAPER/T5/boolq/boolq_virtual/all_results_test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/boolq/boolq_virtual/all_results_test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.txt"
        acc_without_ft = 87.61467889908256
        plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=acc_without_ft, ylabel="Accuracy",
                                title="super_glue_boolq_accuracy (virtual stages)")
        plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="Hours",
                              acc_without_ft=acc_without_ft,
                              ylabel="Accuracy", title="super_glue_boolq_accuracy (virtual stages)")
        print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))  # 2.518
        time_to_best_result(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn)


    def rte_virtual():
        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")
        gpipe_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_goipe_bs_40_se_10_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/rte/rte_virtual/all_results_rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.txt"
        acc_without_ft = 87.72563176895306
        plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=acc_without_ft, ylabel="Accuracy",
                                title="glue_rte_accuracy (virtual stages)")
        plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="Hours",
                              acc_without_ft=acc_without_ft, ylabel="Accuracy",
                              title="glue_rte_accuracy (virtual stages)")
        print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))
        time_to_best_result(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn)


    def rte_seq_hack():
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
        mean_epoch_time = np.mean(d["train_epochs_times"])
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

        mean_epoch_time = np.mean(d["train_epochs_times"])
        best_result_epochs = 51   # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_gpipe = time_to_result
        records.append({"alg": "seq_gpipe",
                        "best_result": 90.97472924187726,
                        "best_result_epoch": 51,
                        "time": time_to_best_stale})

        df = pd.DataFrame.from_records(records)
        print(df)
        speedup_to_best = time_to_best_gpipe / time_to_best_stale
        print("speedup_to_best_result:", speedup_to_best)


    def rte_seq_full():
        # exp_gpipe_fn = "results/t5/glue/rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_gpipe_bs_32_se_32_seed_42.json"
        exp_stale_fn = "results/t5/glue/rte/rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_stale_bs_40_se_10_seed_42.json"
        exp_gpipe_fn = "results/t5/glue/rte/rte_momentum_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_gpipe_bs_40_se_10_seed_42.json"
        # Note stale with with lower micro batch!

        gpipe_fn = "results/FOR_PAPER/T5/rte/rte_seq/all_results_rte_gpipe_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_gpipe_bs_40_se_10_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/rte/rte_seq/all_results_rte_paper_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_t5_tfds_stale_bs_40_se_10_seed_42.txt"

        acc_without_ft = 87.72563176895306

        try:
            plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=acc_without_ft, ylabel="Accuracy",
                                    title="glue_rte_accuracy (seq)")
            plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="Hours",
                                  acc_without_ft=acc_without_ft, ylabel="Accuracy", title="glue_rte_accuracy (seq)")
        except:
            print("failed on some exeption.")
            dump_all_raw_data(exp_stale_fn, exp_gpipe_fn, gpipe_fn, stale_fn, acc_without_ft=acc_without_ft)

            pass
        # print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))
        time_to_best_result(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn)


    def wic_virtual():
        # Note stale with with lower micro batch!
        exp_gpipe_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.json"
        exp_stale_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.json"

        gpipe_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_wic_stale_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.txt"
        acc_without_ft = 72.10031347962382
        plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=acc_without_ft, ylabel="Accuracy",
                                title="super_glue_wic_accuracy (virtual stages)")
        plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="Hours",
                              acc_without_ft=acc_without_ft, ylabel="Accuracy",
                              title="super_glue_wic_accuracy (virtual stages)")
        print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))
        time_to_best_result(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn)


    exps = {
        "boolq": boolq,
        "rte_virtual": rte_virtual,
        "rte_seq": rte_seq_hack,
        "wic_virtual": wic_virtual
    }

    if args.exp in exps:
        exps[args.exp]()
    else:
        raise NotImplementedError(f"exp: {args.exp}, available: {list(exps.keys())}")
    # rte_virtual()
