import ast
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.plot import plot_loss
from experiments.experiments import load_experiment


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


def plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=None, title="super_glue_boolq_accuracy",
                            ylabel=f"Accuracy", checkpoint_every_x_epochs=1, epochs_in_last_checkpoint=None):
    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))

    # change dict keys according to checkpoint_every_x_epochs
    if checkpoint_every_x_epochs > 1:

        gpipe_dict_ = {k * checkpoint_every_x_epochs: v for k, v in list(gpipe_dict.items())[:-1]}
        stale_dict_ = {k * checkpoint_every_x_epochs: v for k, v in list(stale_dict.items())[:-1]}

        if epochs_in_last_checkpoint is None:
            warnings.warn(
                "plot_epochs_vs_accuracy inaccurate point for last epoch, ommiting it. epochs_in_last_checkpoint is not given")
        else:

            k, v = list(gpipe_dict.items())[-1]
            if epochs_in_last_checkpoint == 0:
                gpipe_dict_[k * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v

            else:
                gpipe_dict_[(k - 1) * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v

            k, v = list(stale_dict.items())[-1]
            if epochs_in_last_checkpoint == 0:
                stale_dict_[k * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v
            else:
                stale_dict[(k - 1) * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v

        # TODO...

        gpipe_dict = gpipe_dict_
        stale_dict = stale_dict_

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
                          title="super_glue_boolq_accuracy", ylabel=f"Accuracy", checkpoint_every_x_epochs=1,
                          epochs_in_last_checkpoint=None):
    gpipe_dict, stale_dict, times_gpipe, times_stale = get_fixed_dict_and_times(
        checkpoint_every_x_epochs=checkpoint_every_x_epochs,
        epochs_in_last_checkpoint=epochs_in_last_checkpoint,
        exp_gpipe_fn=exp_gpipe_fn,
        exp_stale_fn=exp_stale_fn,
        gpipe_fn=gpipe_fn, stale_fn=stale_fn,
        time_units=time_units)

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


def get_fixed_dict_and_times(exp_gpipe_fn, exp_stale_fn, gpipe_fn,
                             stale_fn, checkpoint_every_x_epochs=1, epochs_in_last_checkpoint=None, time_units="hours"):
    times_gpipe = extract_times(load_experiment(exp_gpipe_fn), time_units=time_units)
    times_stale = extract_times(load_experiment(exp_stale_fn), time_units=time_units)
    gpipe_dict = extract_values(parse_all_eval_results_dict(gpipe_fn))
    stale_dict = extract_values(parse_all_eval_results_dict(stale_fn))
    # change dict keys according to checkpoint_every_x_epochs
    if checkpoint_every_x_epochs > 1:
        # epochs_in_last
        # gpipe_dict = {k*checkpoint_every_x_epochs: v for k,v in gpipe_dict.items()}
        # stale_dict = {k*checkpoint_every_x_epochs: v for k,v in stale_dict.items()}

        gpipe_dict_ = {k * checkpoint_every_x_epochs: v for k, v in list(gpipe_dict.items())[:-1]}
        stale_dict_ = {k * checkpoint_every_x_epochs: v for k, v in list(stale_dict.items())[:-1]}

        if epochs_in_last_checkpoint is None:
            epochs_in_last_checkpoint = len(times_gpipe) % checkpoint_every_x_epochs
            warnings.warn(
                f"plot_epochs_vs_accuracy may be inaccurate point for last epoch, infering it: epochs_in_last_checkpoint={epochs_in_last_checkpoint}")

        print(f"epochs_in_last_checkpoint={epochs_in_last_checkpoint}")
        k, v = list(gpipe_dict.items())[-1]
        if epochs_in_last_checkpoint == 0:
            gpipe_dict_[k * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v
        else:
            gpipe_dict_[(k - 1) * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v

        k, v = list(stale_dict.items())[-1]
        if epochs_in_last_checkpoint == 0:
            stale_dict_[k * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v
        else:
            stale_dict_[(k - 1) * checkpoint_every_x_epochs + epochs_in_last_checkpoint] = v

        times_gpipe_ = [times_gpipe[i] for i in range(0, len(times_gpipe), checkpoint_every_x_epochs)]
        if len(times_gpipe) % checkpoint_every_x_epochs > 0:
            times_gpipe_.append(times_gpipe[-1])

        times_stale_ = [times_stale[i] for i in range(0, len(times_stale), checkpoint_every_x_epochs)]
        if len(times_stale) % checkpoint_every_x_epochs > 0:
            times_stale_.append(times_stale[-1])

        times_gpipe = times_gpipe_
        times_stale = times_stale_

        stale_dict = stale_dict_
        gpipe_dict = gpipe_dict_
    return gpipe_dict, stale_dict, times_gpipe, times_stale


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


def time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale):
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
                    "best_result_epoch": list(gpipe_dict.keys())[argmax_gpipe],
                    "time": time_to_best_gpipe})
    records.append({"alg": "stale",
                    "best_result": max_stale,
                    "best_result_epoch": list(stale_dict.keys())[argmax_stale],
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

        gpipe_dict, stale_dict, times_gpipe, times_stale = get_fixed_dict_and_times(exp_gpipe_fn, exp_stale_fn,
                                                                                    gpipe_fn, stale_fn)
        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


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
        gpipe_dict, stale_dict, times_gpipe, times_stale = get_fixed_dict_and_times(exp_gpipe_fn, exp_stale_fn,
                                                                                    gpipe_fn, stale_fn)
        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


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
        best_result_epochs = 51  # taken from another exp
        time_to_result = best_result_epochs * mean_epoch_time
        time_to_best_gpipe = time_to_result
        records.append({"alg": "seq_gpipe",
                        "best_result": 90.97472924187726,
                        "best_result_epoch": 51,
                        "time": time_to_best_gpipe})

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

        gpipe_dict, stale_dict, times_gpipe, times_stale = get_fixed_dict_and_times(exp_gpipe_fn, exp_stale_fn,
                                                                                    gpipe_fn, stale_fn)
        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


    def wic_virtual():
        # Note stale with with lower micro batch!
        exp_gpipe_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.json"
        exp_stale_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.json"

        gpipe_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.txt"
        stale_fn = "results/FOR_PAPER/T5/wic/wic_virtual/all_results_wic_stale_test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.txt"
        acc_without_ft = 72.10031347962382
        checkpoint_every_x_epochs = 500 // (5427 // 128)
        plot_epochs_vs_accuracy(gpipe_fn, stale_fn, acc_without_ft=acc_without_ft, ylabel="Accuracy",
                                title="super_glue_wic_accuracy (virtual stages)",
                                checkpoint_every_x_epochs=checkpoint_every_x_epochs)
        plot_time_vs_accuracy(exp_gpipe_fn, exp_stale_fn, gpipe_fn, stale_fn, time_units="Hours",
                              acc_without_ft=acc_without_ft, ylabel="Accuracy",
                              title="super_glue_wic_accuracy (virtual stages)",
                              checkpoint_every_x_epochs=checkpoint_every_x_epochs)
        print("epoch_speedup", epoch_speedup(exp_gpipe_fn, exp_stale_fn))

        gpipe_dict, stale_dict, times_gpipe, times_stale = get_fixed_dict_and_times(exp_gpipe_fn, exp_stale_fn,
                                                                                    gpipe_fn,
                                                                                    stale_fn,
                                                                                    checkpoint_every_x_epochs=checkpoint_every_x_epochs,
                                                                                    epochs_in_last_checkpoint=None,
                                                                                    time_units="hours")

        time_to_best_result(gpipe_dict, stale_dict, times_gpipe, times_stale)


    def wic_seq():
        #results / t5 / super_glue / wic / no_virtual_stages_benchmark_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic_t5_tfds_stale_bs_128_se_4_seed_42.json
        #results/t5/super_glue/wic/no_virtual_stages_benchmark_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_acyclic_t5_tfds_gpipe_bs_128_se_8_seed_42.json

        records = []

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

    def boolq_seq():
        records = []
        d = {"train_epochs_times": [
            3151.963354110718,
            3157.5056524276733
        ],}

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

    def one_loss_plot(fn, legend, fig, step_every):
        config, fit_res = load_experiment(fn)
        loss_per_batch = "loss_per_batch" in config['statistics']
        fig, ax = plot_loss(fit_res, fig=fig, log_loss=False,
                            legend=legend, loss_per_batch=loss_per_batch, step_every=step_every)

        return fig, ax

    def boolq_loss_plots():
        # boolq
        exp_stale_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_20_se_5_seed_42.json"
        exp_gpipe_fn = "results/t5/super_glue/boolq/test_vs_t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_20_se_5_seed_42.json"

        fig, ax = one_loss_plot(fn=exp_stale_fn, legend="stale", fig=None, step_every=10*10)
        fig, ax = one_loss_plot(fn=exp_gpipe_fn, legend="gpipe", fig=fig, step_every=10*10)

        plt.show()

    def wic_loss_plots():
        # wic
        exp_gpipe_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_128_se_8_seed_42.json"
        exp_stale_fn = "results/t5/super_glue/wic/test_vs_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_128_se_2_seed_42.json"

        fig, ax = one_loss_plot(fn=exp_stale_fn, legend="stale", fig=None, step_every=2*10)
        fig, ax = one_loss_plot(fn=exp_gpipe_fn, legend="gpipe", fig=fig, step_every=8*10)

        plt.show()

    def rte_loss_plots():

        exp_results_dir = "results/t5/glue/rte/"
        exp_stale_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_stale_bs_40_se_5_seed_42.json")
        exp_gpipe_fn = os.path.join(exp_results_dir,
                                    "rte_virtual_t5_3b_tied_lmheads_320_8_8p_bw12_squad1_virtual_stages_t5_tfds_gpipe_bs_40_se_10_seed_42.json")

        fig, ax = one_loss_plot(fn=exp_stale_fn, legend="stale", fig=None, step_every=5*10)
        fig, ax = one_loss_plot(fn=exp_gpipe_fn, legend="gpipe", fig=fig, step_every=10*10)

        plt.show()


    exps = {
        "boolq_virtual": boolq_virtual,
        "boolq_seq": boolq_seq,
        "rte_virtual": rte_virtual,
        "rte_seq": rte_seq_hack,
        "wic_virtual": wic_virtual,
        "wic_seq": wic_seq
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
    # rte_virtual()
