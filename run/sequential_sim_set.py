import os
from sklearn.model_selection import ParameterGrid
from .gpu_queue import map_to_limited_gpus, map_to_several_limited_gpus
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
    print(f"-I- Runnning: {command_line}")
    p = subprocess.Popen(args)
    p.wait()


def run_grid_on(COMMAND, param_grid, gpu_list, skip_first=0):
    # Assumes required gpu per run is 1
    configs = ParameterGrid(param_grid)
    if skip_first > 0:
        print(f"-I- Skipping first {skip_first} configs")
        print(f"-I- Skipping: {list(configs)[:skip_first]}")
        configs = list(configs)[skip_first:]
    func = partial(subproccess_func, COMMAND)
    # func = partial(call_function, COMMAND)
    map_to_limited_gpus(func, configs, len(gpu_list),
                        CUDA_VISIBLE_DEVICES=gpu_list)


def run_grid_on_multi_gpu_per_run(COMMAND, param_grid, gpu_list, gpus_per_config=1):
    # TODO: support list gpus_per_run

    # Assumes required gpu per run is 1
    configs = ParameterGrid(param_grid)
    # func = partial(subproccess_func, COMMAND)
    func = partial(call_function, COMMAND)
    map_to_several_limited_gpus(func, configs, gpus_per_config, len(gpu_list),
                                CUDA_VISIBLE_DEVICES=gpu_list)


def infer_number_of_gpus(COMMAND):
    # TODO
    raise NotImplementedError()
