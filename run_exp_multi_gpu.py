from run.sequential_sim_set import run_grid_on_multi_gpu_per_run, RunGridHelper

ALL_SEEDS = [42, 20202020, 77777777, 314159, 1322019]


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
        "pipeline_num_processes": [gpus_per_config],
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=gpus_per_config)


def multiprocessing_cv_debug_step_every(alg="stale_nr", model="wrn_28x10_c100_dr03_p4_group_norm", port=29500, seed=42):
    COMMAND = "python main.py"
    cv_cfgs_dir = "configs/cv/cifar100/wrn28x10/no_recomputation/"
    gpus_per_config = 4
    cfgs_dir = cv_cfgs_dir

    all_algs = [alg]  # TODO
    common = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [seed],
        "pipeline_num_processes": [gpus_per_config],
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
    # param_grid = [param_grid_1]

    for i, p in enumerate(param_grid):
        p.update(**common)
        p["master_port"] = [port+i]


    helper.add(COMMAND, param_grid, gpus_per_config=gpus_per_config)
    
    # run_grid_on_multi_gpu_per_run(COMMAND,
    #                              param_grid,
    #                              gpu_list=list(range(8)),
    #                              gpus_per_config=gpus_per_config)


def mp_gpt2xl_untied():
    COMMAND = "python main.py"
    gpus_per_config = 8
    all_algs = ["stale", "msnag"]  # TODO
    cfgs_dir = "configs/lm/wt2/gpt2xl/untied/"
    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42],
        "pipeline_num_processes": [gpus_per_config],
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
        "pipeline_num_processes": [gpus_per_config],
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=gpus_per_config)

    # python main.py  --config "configs/lm/wt2/gpt2/tied/stale.json" --seed 42 --pipeline_num_processes 5 --debug -1


if __name__ == "__main__":
    global helper
    helper = RunGridHelper(verbose=True, test=False, gpu_list=list(range(8)))
    # gpt2xl_untied()
    # gpt2_tied()
    # gpt2xl_untied_gpipe()
    # mp_gpt2_tied()
    # mp_gpt2xl_untied()  # TODO: bug?
    # gpt2xl_tied()
    multiprocessing_cv_debug_step_every("stale_nr", port=29500, seed=42)
    multiprocessing_cv_debug_step_every("msnag_nr", port=29600, seed=42)

    multiprocessing_cv_debug_step_every("stale_nr", port=29700, seed=1322019)
    multiprocessing_cv_debug_step_every("msnag_nr", port=29800, seed=1322019)
    helper.run()

