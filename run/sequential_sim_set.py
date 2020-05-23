import os
from sklearn.model_selection import ParameterGrid
from .gpu_queue import map_to_limited_gpus, map_to_several_limited_gpus, flexible_map_to_several_limited_gpus
from functools import partial

import shlex
import subprocess


def call_function(COMMAND, *args, _verbose=True, _test=False, **kw):
    """
    Example:
        The following:
            COMMAND = "python main_sequential.py"
            call_function(COMMAND, **dict(seed=42))

        calls:
            python main_sequential.py --seed 42

    """
    sargs = "--" + " --".join([f"{i} {v}" for i, v in kw.items()])
    cmd = f"{COMMAND} {sargs}"
    if _verbose:
        print(cmd)
    if _test:
        return
    os.system(cmd)


def subprocess_func(COMMAND, *args, **kw):
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
    func = partial(subprocess_func, COMMAND)
    # func = partial(call_function, COMMAND)
    map_to_limited_gpus(func, configs, len(gpu_list),
                        CUDA_VISIBLE_DEVICES=gpu_list)


def run_grid_on_multi_gpu_per_run(COMMAND, param_grid, gpu_list, gpus_per_config=1):
    # TODO: support list gpus_per_run

    # Assumes required gpu per run is 1
    configs = ParameterGrid(param_grid)
    # func = partial(subprocess_func, COMMAND)
    func = partial(call_function, COMMAND)
    map_to_several_limited_gpus(func, configs, gpus_per_config, len(gpu_list),
                                CUDA_VISIBLE_DEVICES=gpu_list)


class RunGridHelper:
    def __init__(self, verbose=True, test=False, gpu_list=None):
        self.grids = []
        self.gpu_list = gpu_list if gpu_list else []
        self.verbose = verbose
        self.test = test

    def add(self, COMMAND, param_grid, gpus_per_config):
        func = partial(call_function, COMMAND, _verbose=self.verbose, _test=self.test)
        assert isinstance(gpus_per_config, int)

        def pack_single(g):
            g['FUNC'] = [func]
            g['REQUIRED_GPUS'] = [gpus_per_config]

        if isinstance(param_grid, dict):
            pack_single(param_grid)
            self.grids.append(param_grid)
        else:
            assert isinstance(param_grid, list)
            for g in param_grid:
                pack_single(g)
            self.grids.extend(param_grid)

    def run(self):
        param_grid = self.grids
        configs = ParameterGrid(param_grid)
        gpu_list = self.gpu_list
        flexible_map_to_several_limited_gpus(configs, len(gpu_list), CUDA_VISIBLE_DEVICES=gpu_list)


def infer_number_of_gpus(COMMAND):
    # TODO
    raise NotImplementedError()
