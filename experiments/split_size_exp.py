import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import timeit
from .utils import *


def exp_split_size(model_class, num_devices: int, num_classes: int, batch_shape: Tuple[int, ...], model_params: dict,
                   tests_config: dict, pipeline_params: dict):
    num_repeat = 10

    tests_config['num_classes'] = num_classes
    tests_config['batch_shape'] = batch_shape

    pipeline_params['devices'] = list(range(num_devices))

    stmt = call_func_stmt(train, 'model', **tests_config)

    model_init_stmt = call_func_stmt(model_class, **model_params)

    means = []
    stds = []
    split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

    for split_size in split_sizes:
        pipeline_params['microbatch_size'] = split_size
        setup = f"model = {call_func_stmt(create_pipeline, model_init_stmt, batch_shape, **pipeline_params)}"
        pp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        means.append(np.mean(pp_run_times))
        stds.append(np.std(pp_run_times))
        print(
            f'Split size {split_size} has a mean execution time of {means[-1]} with standard deviation of {stds[-1]}')

    fig, ax = plt.subplots()
    ax.plot(split_sizes, means)
    ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xlabel('Pipeline Split Size')
    ax.set_xticks(split_sizes)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig("split_size_tradeoff.png")
    plt.close(fig)


if __name__ == '__main__':
    parser = ExpParser(uses_dataset=False, description='Run the speedup experiment.')
    args = parser.parse_args()
    exp_split_size(**args)
