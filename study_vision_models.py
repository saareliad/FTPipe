""" This is a skeleton for running bw vs speedup experiments, 
    while overriding some parmeters """
import torch
from pytorch_Gpipe.model_profiling import Node, NodeTypes
import argparse
import importlib
from misc import run_analysis, run_partitions
from pytorch_Gpipe.utils import layerDict, tensorDict
from pytorch_Gpipe import PipelineConfig, pipe_model
from heuristics import node_weight_function, edge_weight_function
# TODO: instea of code copy, do repeated calls to exisitng functions...
from types import SimpleNamespace


class OverrrideArgsDict(SimpleNamespace):
    """ if key in override_dict, return its overriden value,
    else return its value from args """

    def __init__(self, args, override_dict):
        self.__dict__.update(args.__dict__)
        self.__dict__.update(override_dict)


def single_partitioning_loop_with_override(args, METIS_opt, **override_dict):

    # Override variables
    print(f"Overriding: {override_dict}")
    args = OverrrideArgsDict(args, override_dict)

    # Run script as normal...

    # Get model and sample
    # model = create_model(args.model)
    # sample = create_random_sample(args, analysis=False)

    # put them on CPU or GPU according to args.
    # Set heurisitcs
    # graph = pipe_model(model, ...,METIS_opt=METIS_opt)

    # Get Config
    # Run analysis
    # expected_speedup = run_analysis(...)

    # return expected_speedup


def parse_cli():
    raise NotImplementedError()


def bw_vs_speedup_exp():
    args, METIS_opt = parse_cli()
    BW_RANGE = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    expected_speedup = []
    i = 0

    # for bw in BW_RANGE:
    while i < len(BW_RANGE):
        try:
            bw = BW_RANGE[i]
            es = single_partitioning_loop_with_override(args, METIS_opt, bw=bw)
            expected_speedup.append(es)
            i += 1
        except:
            # Failed, try again
            pass

    print('-I- final study results:')
    print("bw", BW_RANGE)
    print("speedup", expected_speedup)

    return BW_RANGE, expected_speedup


if __name__ == "__main__":
    bw_vs_speedup_exp()
