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
    run_grid_on_multi_gpu_per_run(COMMAND, param_grid, gpu_list=list(range(8)), gpus_per_config=4)

if __name__ == "__main__":
    main()
