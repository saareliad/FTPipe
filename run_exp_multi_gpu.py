from run.helper import run_grid_on_multi_gpu_per_run, RunGridHelper
import argparse

ALL_SEEDS = [42, 20202020, 77777777, 314159, 1322019]

# TODO: run replication scripts according to paper, clean.

def gpt2_tied():
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


def gpt2xl_untied():
    COMMAND = "mpirun -np 8 python main.py"
    cfgs_dir = "configs/lm/wt2/gpt2xl/untied/"
    all_algs = ["msnag", "stale", "seq"]
    # all_algs = ["stale", "msnag", "ws", "ws_msnag_ga", "ws_msnag_ga_jfl", "ws_msnag"]
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42, 20202020, 77777777, 314159, 1322019]
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=8)


def gpt2xl_untied_gpipe():
    COMMAND = "mpirun -np 8 python main.py"
    cfgs_dir = "configs/lm/wt2/gpt2xl/untied/"
    all_algs = ["gpipe"]
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        # 'seed': [42, 20202020, 77777777, 314159, 1322019]
        'seed': [42]
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=8)


def gpt2xl_tied():
    COMMAND = "mpirun -np 9 python main.py"
    cfgs_dir = "configs/lm/wt2/gpt2xl/tied/"
    all_algs = ["stale"]
    all_algs = ["seq", "gpipe"]
    # all_algs = ["stale", "msnag", "ws", "ws_msnag_ga", "ws_msnag_ga_jfl", "ws_msnag"]
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42]
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=8)


def multiprocessing_cv():
    COMMAND = "python main.py"
    cv_cfgs_dir = "configs/cv/cifar100/wrn28x10/"
    gpus_per_config = 4
    all_algs = ["stale", "msnag"]  # TODO
    # cfgs_dir = "configs/lm/wt2/gpt2xl/tied/"
    cfgs_dir = cv_cfgs_dir
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42],
        "nprocs": [gpus_per_config],
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=gpus_per_config)



def mp_gpt2xl_untied():
    COMMAND = "python main.py"
    gpus_per_config = 8
    all_algs = ["stale", "msnag"]  # TODO
    cfgs_dir = "configs/lm/wt2/gpt2xl/untied/"
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42],
        "nprocs": [gpus_per_config],
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=gpus_per_config)


def mp_gpt2_tied():
    COMMAND = "python main.py"
    gpus_per_config = 5
    all_algs = ["stale", "msnag"]  # TODO
    cfgs_dir = "configs/lm/wt2/gpt2/tied/"
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42],
        "nprocs": [gpus_per_config],
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=gpus_per_config)

    # python main.py  --config "configs/lm/wt2/gpt2/tied/stale.json" --seed 42 --nprocs 5 --debug -1


def replicate_grad_accumulation_exp_WRN():
    # FIXME: port hack here because we use hardcoded port and we want exps to run in parallel.
    def mp_cv_grad_accumulation(helper,
                                alg="stale_nr",
                                model="wrn_28x10_c100_dr03_p4_group_norm",
                                port=29500,
                                seed=42):
        COMMAND = "python main.py"
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


#######################################################
AVAIALBE_EXPS = {'replicate_grad_accumulation_exp_WRN': replicate_grad_accumulation_exp_WRN}


def parse_cli():
    # TODO:
    parser = argparse.ArgumentParser("replicate experiments grid")
    parser.add_argument('-e', "--exp", choices=AVAIALBE_EXPS.keys(), default='replicate_grad_accumulation_exp_WRN')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_cli()
    fn = AVAIALBE_EXPS[args.exp]
    fn()
    # replicate_grad_accumulation_exp_WRN()


    # gpt2xl_untied()
    # gpt2_tied()
    # gpt2xl_untied_gpipe()
    # mp_gpt2_tied()
    # mp_gpt2xl_untied()  # TODO: bug?
    # gpt2xl_tied()


