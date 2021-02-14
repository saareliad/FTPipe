import argparse

from pipe.run.helper import run_grid_on_multi_gpu_per_run, RunGridHelper

ALL_SEEDS = [42, 20202020, 77777777, 314159, 1322019]


# TODO: run replication scripts according to paper, clean.
# TODO: can re run with better partitioning when we compare to GPipe.


def gpt2_tied():
    # TODO:
    # this will be vs the untied version. The goal is showing high utilization and reporting it.
    # can also check it achieves simillar accuracy.
    # NOTE: mpi is needed.
    COMMAND = "mpirun -np 5 python main.py"
    cfgs_dir = "configs/lm/wt2/gpt2/tied/"
    all_algs = ["stale"]
    # all_algs = ["stale", "seq", "ws", "msnag", "ws_msnag_ga", "ws_msnag_ga_jfl", "ws_msnag"]

    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        # 'seed': [42, 20202020, 77777777, 314159, 1322019]
        'seed': [1322019]
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=4)


################################################################


def gpt2xl():
    # Goal: achieving comparable generalization vs fully-synchronous models
    # NOTE: also better than stale, also better than aggmsnag.
    # NOTE: increasing the batch (more micro batches) give lower generalization.
    # NOTE: lower sequence length give lower generalization
    COMMAND = "python main.py --mode mp --nprocs 8 --step_every 8 --step_every_from_cmd"
    cfgs_dir = "configs/lm/wt2/gpt2xl/untied/"
    all_algs = ["aggmsnag", "stale", "seq", "gpipe"]
    # all_algs = ["stale", "msnag", "aggmsnag", "ws", "ws_msnag_ga", "ws_msnag_ga_jfl", "ws_msnag"]
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42, 20202020, 77777777, 314159, 1322019]
    }
    gpus_per_config = 8
    helper = RunGridHelper(verbose=True, test=False, gpu_list=list(range(8)))
    helper.add_runs(COMMAND, param_grid, num_gpus=gpus_per_config)
    helper.run()
    #
    # run_grid_on_multi_gpu_per_run(COMMAND,
    #                               param_grid,
    #                               gpu_list=list(range(8)),
    #                               gpus_per_config=8)


def grad_accumulation_WRN():
    # Goal: show grad accumulation dramatically improves results.
    # Weight prediction is also an improvement compared to stale.
    # TODO: weight prediction with aggregation (aggmsnag_nr)
    # NOTE: these exps run without re-computation. recomputation didn't change results

    # HACK: port hack here because we use hardcoded port and we want exps to run in parallel.
    def mp_cv_grad_accumulation(helper,
                                alg="stale_nr",
                                model="wrn_28x10_c100_dr03_p4_group_norm",
                                port=29500,
                                seed=42):
        COMMAND = "python main.py --mode mp"
        cv_cfgs_dir = "configs/cv/cifar100/wrn28x10/no_recomputation/"
        gpus_per_config = 4
        cfgs_dir = cv_cfgs_dir

        all_algs = [alg]  # TODO
        common = {
            'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
            'seed': [seed],
            "nprocs": [gpus_per_config],
            "step_every_from_cmd": [""],
            "bs_train_from_cmd": [""],
            "out_dir_from_cmd": [""],
            "out_dir": ["results/debug_se/gn/linscale/"],
            "model": [model],
            "model_from_cmd": [""]
        }
        param_grid_1 = {
            "step_every": [1],
            "bs_train": [256],
        }

        param_grid_2 = {
            "step_every": [2],
            "bs_train": [128],
        }

        param_grid_3 = {
            "step_every": [4],
            "bs_train": [64],
        }

        param_grid = [param_grid_1, param_grid_2, param_grid_3]

        for i, p in enumerate(param_grid):
            p.update(**common)
            p["master_port"] = [port + i]

        helper.add_runs(COMMAND, param_grid, num_gpus=gpus_per_config)

    helper = RunGridHelper(verbose=True, test=False, gpu_list=list(range(8)))
    mp_cv_grad_accumulation(helper, "stale_nr", port=29500, seed=42)
    mp_cv_grad_accumulation(helper, "msnag_nr", port=29600, seed=42)

    mp_cv_grad_accumulation(helper, "stale_nr", port=29700, seed=1322019)
    mp_cv_grad_accumulation(helper, "msnag_nr", port=29800, seed=1322019)
    helper.run()


def t5_glue():
    ALL_TASKS = {

    }

    def mp_helper(helper,
                  alg="stale_nr",
                  model="wrn_28x10_c100_dr03_p4_group_norm",
                  port=29500,
                  seed=42):
        COMMAND = "python main.py --mode mp"
        cv_cfgs_dir = "configs/cv/cifar100/wrn28x10/no_recomputation/"
        gpus_per_config = 4
        cfgs_dir = cv_cfgs_dir

        all_algs = [alg]  # TODO
        common = {
            'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
            'seed': [seed],
            "nprocs": [gpus_per_config],
            "step_every_from_cmd": [""],
            "bs_train_from_cmd": [""],
            "out_dir_from_cmd": [""],
            "out_dir": ["results/debug_se/gn/linscale/"],
            "model": [model],
            "model_from_cmd": [""]
        }
        param_grid_1 = {
            "step_every": [1],
            "bs_train": [256],
        }

        param_grid_2 = {
            "step_every": [2],
            "bs_train": [128],
        }

        param_grid_3 = {
            "step_every": [4],
            "bs_train": [64],
        }

        param_grid = [param_grid_1, param_grid_2, param_grid_3]

        for i, p in enumerate(param_grid):
            p.update(**common)
            p["master_port"] = [port + i]

        helper.add_runs(COMMAND, param_grid, num_gpus=gpus_per_config)


#######################################################

AVAIALBE_EXPS = {'grad_accumulation_WRN': grad_accumulation_WRN,
                 'gpt2xl': gpt2xl}


def parse_cli():
    # TODO:
    parser = argparse.ArgumentParser("replicate experiments grid")
    parser.add_argument('-e', "--exp", choices=AVAIALBE_EXPS.keys(), default='grad_accumulation_WRN')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()
    fn = AVAIALBE_EXPS[args.exp]
    fn()
    # grad_accumulation_WRN()
    # gpt2xl()

    # gpt2_tied()
    # gpt2xl_untied_gpipe()
    # mp_gpt2_tied()
    # mp_gpt2xl_untied()  # TODO: bug?
    # gpt2xl_tied()
