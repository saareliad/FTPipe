from run.sequential_sim_set import run_grid_on_multi_gpu_per_run

ALL_SEEDS = [42, 20202020, 77777777, 314159, 1322019]


def main():
    COMMAND = "mpirun -np 5 python main.py"
    cfgs_dir = "configs/lm/wt2/gpt2/tied_weights/"
    all_algs = ["seq", "stale"]

    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [42]
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=4)


def main2():
    COMMAND = "mpirun -np 5 python main.py"
    cfgs_dir = "configs/lm/wt2/gpt2/tied_weights/"
    all_algs = ["stale"]

    param_grid = {
        'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
        'seed': [20202020, 77777777, 314159, 1322019]
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
        # 'seed': [42, 20202020, 77777777, 314159, 1322019]
        'seed': [42, 20202020, 77777777, 314159, 1322019]
    }
    run_grid_on_multi_gpu_per_run(COMMAND,
                                  param_grid,
                                  gpu_list=list(range(8)),
                                  gpus_per_config=8)

if __name__ == "__main__":
    gpt2xl_untied()
    # main2()
