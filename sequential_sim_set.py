import os
from sklearn.model_selection import ParameterGrid
from gpu_queue import map_to_limited_gpus, map_to_several_limited_gpus
from functools import partial

import shlex
import subprocess


def call_function(COMMAND, *args, **kw):
    """
    Example:
        The following:
            COMMAND = "python main_sequential.py"
            call_function(COMMAND, **dict(seed=42))

        calls:
            python main_sequential.py --seed 42

    """
    sargs = "--" + " --".join([f"{i} {v}" for i, v in kw.items()])
    os.system(f"{COMMAND} {sargs}")


def subproccess_func(COMMAND, *args, **kw):
    sargs = "--" + " --".join([f"{i} {v}" for i, v in kw.items()])
    command_line = f"{COMMAND} {sargs}"
    args = shlex.split(command_line)
    p = subprocess.Popen(args)
    p.wait()


def run_grid_on(COMMAND, param_grid, gpu_list):
    # Assumes required gpu per run is 1
    configs = ParameterGrid(param_grid)
    func = partial(subproccess_func, COMMAND)
    # func = partial(call_function, COMMAND)
    map_to_limited_gpus(func, configs, len(gpu_list),
                        CUDA_VISIBLE_DEVICES=gpu_list)


def run_grid_on_multi_gpu_per_run(COMMAND, param_grid, gpu_list, gpus_per_config=1):
    # TODO: support list gpus_per_run

    # Assumes required gpu per run is 1
    configs = ParameterGrid(param_grid)
    func = partial(subproccess_func, COMMAND)
    # func = partial(call_function, COMMAND)
    map_to_several_limited_gpus(func, configs, gpus_per_config, len(gpu_list),
                                CUDA_VISIBLE_DEVICES=gpu_list)


def infer_number_of_gpus(COMMAND):
    # TODO
    raise NotImplementedError()


if __name__ == "__main__":
    def sequential():
        COMMAND = "python main_sequential.py"
        param_grid = {
            'config': [
                # 'configs/sequential/seq_wrn16x4_c10.json',
                'configs/sequential/seq_wrn16x4_c100.json',
                # 'config': ['configs/sequential/seq_wrn28x10_c100.json'
            ],
            'seed': [42, 20202020, 77777777, 314159, 1322019]
        }
        run_grid_on(COMMAND, param_grid, gpu_list=[0, 1, 2, 3, 4, 5])

    def staleness_study():
        COMMAND = "mpirun -np 4 python main.py"
        cfgs_dir = "configs/wrn16x4_cifar100/"
        all_algs = ["weight_stashing_msnag_gap_aware",
                    "weight_stashing",
                    "weight_stashing_msnag",
                    "weight_stashing_gap_aware",
                    "gap_aware",
                    "msnag",
                    "stale", ]

        param_grid = {
            'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
            'seed': [42, 20202020, 77777777, 314159, 1322019]
        }
        run_grid_on(COMMAND, param_grid, gpu_list=[0, 1, 2, 3, 4, 5])

    def sim_ddp4():
        COMMAND = "python main_sequential.py"
        cfgs_dir = "configs/ddp_sim/ddp_4gpus/"
        all_algs = ["seq_wrn16x4_c100",
                    "seq_wrn16x4_c10",
                    "seq_wrn28x10_c100"]
        # TODO: seq_wrn28x10_c100 did not run due to insufficient memory.

        param_grid = {
            'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
            'seed': [42, 20202020, 77777777, 314159, 1322019]
        }
        run_grid_on(COMMAND, param_grid, gpu_list=[0, 1, 2, 3, 4, 5])

    def sim_ddp8():
        COMMAND = "python main_sequential.py"
        cfgs_dir = "configs/ddp_sim/ddp_8gpus/"
        all_algs = ["seq_wrn16x4_c100",
                    "seq_wrn16x4_c10",
                    # "seq_wrn28x10_c100"
                    ]
        # TODO: seq_wrn28x10_c100 did not run due to insufficient memory.

        param_grid = {
            'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
            'seed': [42, 20202020, 77777777, 314159, 1322019]
        }
        run_grid_on(COMMAND, param_grid, gpu_list=[0, 1, 2, 3, 4, 5])

    def ddp4():
        COMMAND = "numactl --cpunodebind=0 python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 main_sequential.py"
        # --master_port 6001
        cfgs_dir = "configs/ddp/"
        all_algs = [  # "seq_wrn16x4_c100",
            # "seq_wrn16x4_c10",
            "seq_wrn28x10_c100"]
        # TODO: seq_wrn28x10_c100 did not run due to insufficient memory.

        param_grid = {
            'config': [f"{cfgs_dir}{cfg}.json" for cfg in all_algs],
            'seed': [42, 20202020, 77777777, 314159, 1322019]
        }
        run_grid_on_multi_gpu_per_run(COMMAND, param_grid, gpu_list=[0, 1, 2, 3, 4, 5], gpus_per_config=4)

    ddp4()
