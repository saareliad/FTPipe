import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import timeit
from sample_models import *
from .utils import *

plt.switch_backend('Agg')


def exp_model_time(run_type, model_class, num_classes, batch_shape: Tuple[int, ...], num_repeats, num_warmups,
                   model_params: dict, tests_config: dict, pipeline_params: dict = None, num_devices=None):

    tests_config['num_classes'] = num_classes
    tests_config['batch_shape'] = batch_shape

    stmt = call_func_stmt(train, 'model', **tests_config)

    model_init_stmt = call_func_stmt(model_class, **model_params)

    device_str = "'cuda:0' if torch.cuda.is_available() else 'cpu'"

    if run_type in ['S', 'Single']:
        setup = f"model = {model_init_stmt}.to({device_str})"

        run_times = timeit.repeat(stmt, setup, number=1, repeat=num_warmups + num_repeats, globals=globals())

        rt_mean, rt_std = np.mean(run_times[num_warmups:]), np.std(run_times[num_warmups:])
        max_mem = get_max_memory_allocated()
        print('Single GPU:')

    elif run_type in ['P', 'Pipeline-Parallel']:
        pipeline_params['devices'] = list(range(num_devices))
        setup = f"model = {call_func_stmt(create_pipeline, model_init_stmt, batch_shape, **pipeline_params)}"

        run_times = timeit.repeat(stmt, setup, number=1, repeat=num_warmups + num_repeats, globals=globals())

        rt_mean, rt_std = np.mean(run_times[num_warmups:]), np.std(run_times[num_warmups:])
        max_mem = get_max_memory_allocated()
        print('Pipeline-Parallel:')

    elif run_type in ['D', 'Data-Parallel']:
        devices_ids = list(range(num_devices))
        setup = f"model = nn.DataParallel({model_init_stmt}, device_ids={devices_ids}).to({device_str})"

        dp_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_warmups + num_repeats, globals=globals())

        rt_mean, rt_std = np.mean(dp_run_times[num_warmups:]), np.std(dp_run_times[num_warmups:])
        max_mem = get_max_memory_allocated()
        print('Data-Parallel:')

    else:
        raise ValueError('Not a valid run type')

    print(f'\trun time mean - {rt_mean}')
    print(f'\trun time std - {rt_std}')
    print(f'\tmax memory usage - {max_mem}')


if __name__ == '__main__':
    parser = ExpParser(uses_dataset=False, description='Run the speedup experiment.')
    args = parser.parse_args()
    exp_model_time(**args)
