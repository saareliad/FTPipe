import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import timeit
from .utils import *

plt.switch_backend('Agg')


def exp_model_time(model_class, num_devices: int, num_classes: int, batch_shape: Tuple[int, ...], model_params: dict,
                   tests_config: dict, pipeline_params: dict):
    num_repeat = 10

    tests_config['num_classes'] = num_classes
    tests_config['batch_shape'] = batch_shape

    model_params['num_classes'] = num_classes

    pipeline_params['devices'] = list(range(num_devices))

    stmt = call_func_stmt(train, 'model', **tests_config)

    model_init_stmt = call_func_stmt(model_class, **model_params)

    device_str = """'cuda:0' if torch.cuda.is_available() else 'cpu'"""

    setup = f"model = {model_init_stmt}.to({device_str})"
    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

    print('finished single gpu')

    setup = f"model = {call_func_stmt(create_pipline, model_init_stmt, batch_shape, **pipeline_params)}"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    print('finished pipeline')

    plot([mp_mean, rn_mean],
         [mp_std, rn_std],
         ['Model Parallel', 'Single GPU'],
         'mp_vs_rn.png')

    if torch.cuda.is_available():
        setup = f"model = nn.DataParallel({model_init_stmt}).to({device_str})"
        dp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        dp_mean, dp_std = np.mean(dp_run_times), np.std(dp_run_times)
        print(f'Data parallel mean is {dp_mean}')

        print(
            f'data parallel has speedup of {(rn_mean / dp_mean - 1) * 100}% relative to single gpu')
        plot([mp_mean, rn_mean, dp_mean],
             [mp_std, rn_std, dp_std],
             ['Model Parallel', 'Single GPU', 'Data Parallel'],
             'mp_vs_rn_vs_dp.png', 'ResNet50 Execution Time (Second)')

    print(
        f'pipeline has speedup of {(rn_mean / mp_mean - 1) * 100}% relative to single gpu')
    assert mp_mean < rn_mean
    # assert that the speedup is at least 30%
    assert rn_mean / mp_mean - 1 >= 0.3
