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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if run_type in ['S', 'Single']:
        tests_config['model'] = model_class(**model_params).to(device)

        print('Single GPU:')

    elif run_type in ['P', 'Pipeline-Parallel']:
        pipeline_params['devices'] = list(range(num_devices))
        model = model_class(**model_params).to(device)
        tests_config['model'] = create_pipeline(model, batch_shape, **pipeline_params)

        print('Pipeline-Parallel:')

    elif run_type in ['D', 'Data-Parallel']:
        devices_ids = list(range(num_devices))
        model = model_class(**model_params).to(device)
        tests_config['model'] = nn.DataParallel(model, device_ids=devices_ids).to(device)

        print('Data-Parallel:')

    else:
        raise ValueError('Not a valid run type')

    run_times, mem_uses = track_train(num_repeats + num_warmups, **tests_config)
    run_times, mem_uses = run_times[num_warmups:], mem_uses[num_warmups:]
    rt_mean, rt_std = np.mean(run_times), np.std(run_times)
    max_mem = np.mean(mem_uses)

    print(f'\trun time mean - {rt_mean}')
    print(f'\trun time std - {rt_std}')
    print(f'\tmax memory usage - {max_mem}')


if __name__ == '__main__':
    parser = ExpParser(uses_dataset=False, description='Run the speedup experiment.')
    args = parser.parse_args()
    exp_model_time(**args)
