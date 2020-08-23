
import json
from typing import NamedTuple, Dict
import os
from types import SimpleNamespace


def load_experiment(filename):
    """ Returns:
            config, fit_res
    """
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = output['results']

    return config, fit_res


def load_experiment_for_update(run_name, out_dir):
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    with open(output_filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = output['results']

    return config, fit_res


def save_experiment(run_name, out_dir, config, fit_res: Dict):
    if isinstance(fit_res, NamedTuple):
        fit_res = fit_res._asdict()
    elif isinstance(fit_res, SimpleNamespace):
        fit_res = fit_res.__dict__
    # elif isinstance(fit_res, dict):
    #     pass

    output = dict(
        config=config,
        results=fit_res
    )
    # TODO: add option that file name will not be by run name.
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        try:
            json.dump(output, f, indent=2)
        except Exception as e:
            print("-E- error saving experiment, printing for easier debug")
            print("-"*40)
            print(output)
            print("-"*40)
            raise e

    print(f'*** Output file {output_filename} written')


class ArgsStasher:
    """
    used for naming and reproducibility conventions,
    (as sometimes we change the args inplace)
    """
    STASH_NAME = "_tmp_stashed"

    @staticmethod
    def stash_to_args(args, replaced_key, old_value):
        # make sure this specific replacement does not ruin experiment name
        if not hasattr(args, "auto_file_name"):
            return

        # replaced_key : Union[str, Tuple[str]], List[str]],
        STASH_NAME = ArgsStasher.STASH_NAME
        if isinstance(replaced_key, list):
            replaced_key = tuple(replaced_key)
        if not hasattr(args, STASH_NAME):
            setattr(args, STASH_NAME, dict())
        sd = getattr(args, STASH_NAME)
        sd[replaced_key] = old_value

    @staticmethod
    def reload_stashed_args(args):
        STASH_NAME = ArgsStasher.STASH_NAME
        if not hasattr(args, STASH_NAME):
            return
        sd = getattr(args, STASH_NAME)
        for replaced_key,old_value in sd.items():
            attr = args
            if isinstance(replaced_key, tuple):
                for a in replaced_key[:-1]:
                    attr = getattr(attr, a)
                last_key = replaced_key[-1]
            else:
                last_key = replaced_key

            assert isinstance(last_key, str)
            setattr(attr, last_key, old_value)
        delattr(args, STASH_NAME)


def auto_file_name(args, verbose=True):
    ArgsStasher.reload_stashed_args(args)
    """This is used to distinguish different configurations by file name """
    assert hasattr(args, "auto_file_name")
    wp = args.weight_prediction['type'] if hasattr(
        args, "weight_prediction") else 'stale'
    ws = "ws_" if getattr(args, "weight_stashing", False) else ""
    ga = "ga_" if hasattr(args, "gap_aware") else ""
    bs = f"bs_{args.bs_train * args.step_every}"
    se = f"se_{args.step_every}"
    ga_just_for_loss = "gaJFL_" if getattr(args, 'gap_aware_just_loss',
                                           False) else ""

    if 'gpipe' == args.work_scheduler.lower():
        s = f'{args.model}_{args.dataset}_gpipe_{bs}_{se}_seed_{args.seed}'
    else:
        s = f'{args.model}_{args.dataset}_{wp}_{ws}{ga}{bs}_{se}_{ga_just_for_loss}seed_{args.seed}'
    args.out_filename = f"{args.out_filename}_{s}"
    if verbose:
        print(f"Out File Name will be: {args.out_filename}")
    return args.out_filename