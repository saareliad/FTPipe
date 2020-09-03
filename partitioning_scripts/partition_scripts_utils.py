import os
import shlex
import sys
from shutil import copyfile, rmtree
from typing import Tuple

import torch


def choose_blocks(model, args) -> Tuple[torch.nn.Module]:
    blocks = dict()

    for m in model.modules():
        m_superclasses = {c.__name__: c for c in type(m).mro()}
        blocks.update(m_superclasses)

    if args.basic_blocks is None:
        args.basic_blocks = []
    try:
        return tuple([blocks[name] for name in args.basic_blocks])
    except KeyError:
        raise ValueError(f"invalid basic blocks possible blocks are {list(blocks.keys())}")


def record_cmdline(output_file):
    """Add cmdline to generated python output file."""
    cmdline = " ".join(map(shlex.quote, sys.argv[:]))
    python_output_file = output_file + ".py"
    cmdline = '"""' + "AutoGenerated with:\n" + "python " + cmdline + "\n" + '"""'
    with open(python_output_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(cmdline.rstrip('\r\n') + '\n' + content)


def bruteforce_main(main, main_kwargs=None, override_dicts=None, NUM_RUNS=2, TMP="/tmp/partitioning_outputs/"):
    # TODO: put all hyper parameters here, a dict for each setting we want to try.
    # d1 = dict(basic_blocks=[])
    # ovverride_dicts.append(d1)
    if main_kwargs is None:
        main_kwargs = dict()

    results = {}
    best = None

    if override_dicts is None:
        override_dicts = []

    if not override_dicts:
        override_dicts = [{}]

    os.makedirs(TMP, exist_ok=True)
    DICT_PREFIX = "_d%d"
    current_dict_prefix = ""
    last_exception = None
    for i, override_dict in enumerate(override_dicts):
        if i > 0:
            current_dict_prefix = DICT_PREFIX.format(i)
        for counter in range(NUM_RUNS):
            main_kwargs['override_dict'] = override_dict
            try:
                out = main(**main_kwargs)
            except (Exception, RuntimeError, AssertionError) as e:
                last_exception = e
                continue
            (analysis_result, output_file) = out

            name = output_file
            orig_name = name
            flag = False

            if name in results:
                if name.endswith(".py"):
                    name = name[:-3]
                flag = True

            while (name + ".py" in results) or flag:
                flag = False
                name += f"_{counter}"

            name += current_dict_prefix
            new_path = os.path.join(TMP, name + ".py")
            copyfile(orig_name + ".py", new_path)  # Save the last generated file

            results[name] = analysis_result

            if best is None:
                best = (new_path, analysis_result)
            elif analysis_result > best[1]:
                best = (new_path, analysis_result)

    print(f"best: {best}")
    if best is None:
        print("-I- hyper parameter search failed raising last exception")
        raise last_exception
    copyfile(best[0], orig_name + ".py")
    print(f"-I- copied best to {orig_name}.py")
    rmtree(TMP)
