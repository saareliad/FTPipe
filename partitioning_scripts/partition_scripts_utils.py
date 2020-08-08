import shlex
import sys
from typing import Tuple
import os
from shutil import copyfile

import torch



def choose_blocks(model, args)->Tuple[torch.nn.Module]:
    blocks = dict()

    for m in model.modules():
        m_superclasses = {c.__name__:c for c in type(m).mro()}
        blocks.update(m_superclasses)

    if args.basic_blocks is None:
        args.basic_blocks = []
    return tuple([blocks[name] for name in args.basic_blocks])


def record_cmdline(output_file):
    """Add cmdline to generated python output file."""
    cmdline = " ".join(map(shlex.quote, sys.argv[:]))
    python_output_file = output_file + ".py"
    cmdline = '"""' + "AutoGenerated with:\n" + "python " + cmdline + "\n" + '"""'
    with open(python_output_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(cmdline.rstrip('\r\n') + '\n' + content)


def bruteforce_main(main, override_dicts=None, NUM_RUNS=2, TMP="/tmp/partitioning_outputs/") :
    # TODO: put all hyper parameters here, a dict for each setting we want to try.
    # d1 = dict(basic_blocks=[])
    # ovverride_dicts.append(d1)
    results = {}
    best = None
    
    if override_dicts is None:
        override_dicts = []

    if not override_dicts:
        override_dicts = [{}]

    DICT_PREFIX = "_d%d"
    current_dict_prefix = ""
    for i, override_dict in enumerate(override_dicts):
        if i > 0:
            current_dict_prefix = DICT_PREFIX.format(i)

        counter = 0
        while counter < NUM_RUNS:
            out = main(override_dict)
     
            try:
                os.makedirs(TMP, exist_ok=True)
                (analysis_result, args) = out

                name = args.output_file
                orig_name = name
                flag = False

                if name in results:
                    if name.endswith(".py"):
                        name = name[:-3]
                    flag = True

                while (name+".py" in results) or flag:
                    flag = False
                    name += f"_{counter}"
                
                name += current_dict_prefix
                new_path = os.path.join(TMP, name+".py")
                copyfile(orig_name+".py",  new_path)  # Save the last generated file

                results[name] = analysis_result

                if best is None:
                    best = (new_path, analysis_result)
                else:
                    if analysis_result > best[1]:
                        best = (new_path, analysis_result)

            except Exception as e:
                print("-E- running multiple times failed")
                raise e

            counter += 1

    print(results)
    print(f"best: {best}")
    copyfile(os.path.join(TMP, best[0]), orig_name+".py")
    print(f"-I- copied best to {orig_name}.py")

