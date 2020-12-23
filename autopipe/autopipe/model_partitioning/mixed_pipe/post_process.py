import os

import torch

from autopipe.autopipe.model_partitioning.utils import re_assign_partition_indices, has_stage_cycles
from autopipe.autopipe.model_profiling import Graph

__all__ = ["post_process_partition"]


def post_process_partition(graph: Graph, edge_weight_function=None, verbose_on_error=True,
                           assert_output_types=False) -> Graph:
    """
    process the partition and optimize it
    called as part of partition_graph method

    Parameters:
    ----------
    graph:
        the Graph object that was partitioned
    verbose_on_error:
        print extra info when cycle can't be solved
    """

    # TODO: assert that stages are connected components.

    re_assign_partition_indices(graph)
    # this is a sanity check

    if has_stage_cycles(graph):
        if os.environ.get("DEBUG", False):
            graph.save_as_pdf(f"{graph.model_name}_before_fix",
                              ".")

        problems, info = get_problematic_partitions(graph)
        for p, i in zip(problems, info):
            print("===Problem===")
            print(p)
            for ii in i:
                print(ii)

        error = "error cycle detected mutual dependency between partitions"
        raise AssertionError(error)

    check_partition_outputs(graph, assert_output_types=assert_output_types, edge_weight_function=edge_weight_function)

    return graph


def check_partition_outputs(graph, assert_output_types=False, edge_weight_function=None):
    is_valid, error = is_output_only_tensors(graph, edge_weight_function)
    if assert_output_types:
        assert is_valid, error
    else:
        if not is_valid:
            print("Output between partitions is tricky, but allowing this")
            print_all_problematic_outputs_between_partitions(graph, edge_weight_function)


def get_problematic_partitions(graph):
    """ For debug when cycle are detected """
    problems = []
    info = []
    for u in graph.nodes:
        for v in u.out_edges:
            if v.stage_id < u.stage_id:
                problems.append([u.stage_id, v.stage_id])
                info.append([(u.id, u.stage_id, u.scope), (v.id, v.stage_id, v.scope), ])
    return problems, info


def is_output_only_tensors(graph: Graph, edge_weight_function=None):
    """
    check if we only send tensors between partitions
    """
    for n in graph.nodes:
        if n.value_type in {type(None), list, tuple, dict, set, int, bool, float, str, slice, torch.Size, torch.dtype}:
            for o in n.out_edges:
                if n.stage_id != o.stage_id:
                    msg = f"invalid output type at partition boundary {n.stage_id}=>{o.stage_id}"
                    msg += f"\noutput is {n.scope} of type {n.value_type}"
                    if edge_weight_function is not None:
                        msg += f" weight {edge_weight_function(n, o)}"
                    return False, msg

    return True, ""


def print_all_problematic_outputs_between_partitions(graph: Graph, edge_weight_function=None):
    """
    check if we only send tensors between partitions
    """
    problems = []
    valid_state = True
    for n in graph.nodes:
        if n.value_type in {type(None), list, tuple, dict, set, int, bool, float, str, slice, torch.Size, torch.dtype}:
            for o in n.out_edges:
                if n.stage_id != o.stage_id:
                    msg = f"invalid output type at partition boundary {n.stage_id}=>{o.stage_id}"
                    msg += f"\noutput is {n.scope} of type {n.value_type}"
                    if edge_weight_function is not None:
                        msg += f" weight {edge_weight_function(n, o)}"
                    valid_state = False
                    problems.append(msg)

    s = f"Valid outputs states = {valid_state}\n" + "problems:\n" + "\n".join(problems)
    print(s)
