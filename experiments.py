
import json
from typing import NamedTuple
import os
from types import SimpleNamespace


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = output['results']

    return config, fit_res


def save_experiment(run_name, out_dir, config, fit_res):

    if isinstance(fit_res, NamedTuple):
        fit_res = fit_res._asdict()
    elif isinstance(fit_res, SimpleNamespace):
        fit_res = fit_res.__dict__

    output = dict(
        config=config,
        results=fit_res
    )
    # TODO: add option that file name will not be by run name.
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')
