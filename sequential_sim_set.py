import os
from sklearn.model_selection import ParameterGrid
from gpu_queue import map_to_limited_gpus
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


if __name__ == "__main__":
    COMMAND = "python main_sequential.py"
    param_grid = {
        'config': ['configs/sequential/sequential.json'],
        'seed': [42, 20202020, 77777777, 314159, 1322019]
    }
    run_grid_on(COMMAND, param_grid, gpu_list=[1, 2, 3, 4, 5])