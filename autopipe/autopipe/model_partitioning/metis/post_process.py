import os
from collections import defaultdict
from copy import copy
from typing import Set

import torch

from autopipe.autopipe.model_partitioning.utils import re_assign_partition_indices, has_stage_cycles
from autopipe.autopipe.model_profiling import Graph, Node, NodeTypes

__all__ = ["post_process_partition"]


def post_process_partition(graph: Graph, edge_weight_function, verbose_on_error=True,
                           assert_output_types=False) -> Graph:
    '''
    process the partition and optimize it
    called as part of partition_graph method

    Parameters:
    ----------
    graph:
        the Graph object that was partitioned
    verbose_on_error:
        print extra info when cycle can't be solved
    '''

    re_assign_partition_indices(graph)
    if has_stage_cycles(graph):
        if os.environ.get("DEBUG", False):
            graph.save_as_pdf(f"{graph.model_name}_before_fix",
                              ".")

        break_partition_cycles(graph)

        # possibly redundent
        try:
            re_assign_partition_indices(graph)
        except:
            print(
                "-W- ignoring exception of redundent re_assign_partition_indices(graph)")

    # this is a sanity check
    if has_stage_cycles(graph):
        if os.environ.get("DEBUG", False):
            graph.save_as_pdf(f"{graph.model_name}_after_fix",
                              ".")

        if verbose_on_error:
            problems, info = get_problematic_partitions(graph)
            print("-V- printing problematic partitions")
            for p, i in zip(problems, info):
                print(p)
                print(i)

            n_partitions = len(set(u.stage_id for u in graph.nodes))
            print("n_partitions:", n_partitions)

        error = "error cycle detected mutual dependency between partitions"
        raise AssertionError(error)

    is_valid, error = is_valid_partitioning(graph, edge_weight_function)
    if assert_output_types:
        assert is_valid, error
    else:
        if not is_valid:
            print("Output between partitions is tricky, but allowing this")
            print_all_problematic_outputs_between_partitions(graph, edge_weight_function)

    return graph


def get_problematic_partitions(graph):
    """ For debug when cycle are detected """
    problems = []
    info = []
    for u in graph.nodes:
        for v in u.out_edges:
            if v.stage_id < u.stage_id:
                problems.append([v.stage_id, u.stage_id])
                info.append([v.scope, u.scope])
    return problems, info


def break_partition_cycles(graph: Graph):
    parts = set()
    roots = defaultdict(set)
    # roots[i] = nodes in partition j s.t j<i and exists backward edge from partition i to j
    for u in graph.nodes:
        parts.add(u.stage_id)
        for v in u.out_edges:
            if u.stage_id > v.stage_id:
                roots[v.stage_id].add(v)

    n_parts = len(parts)
    for idx, group in roots.items():
        # each group represents a new partition to create
        for n in find_subtree(group, len(graph.nodes)):
            n.stage_id = n_parts
        n_parts += 1


def find_subtree(roots: Set[Node], graph_size: int):
    nodes = set()
    open = copy(roots)
    while len(open) > 0:
        n = open.pop()
        nodes.add(n)
        for u in n.out_edges:
            if u.stage_id == n.stage_id:
                nodes.add(u)
                open.add(u)

    open = copy(nodes)
    while len(open) > 0:
        n = open.pop()
        if n in roots:
            continue

        for u in n.in_edges:
            if u.stage_id == n.stage_id:
                # TODO we need to know if u is part of the sub tree
                # this is an reasonable estimation
                if u.type != NodeTypes.IN and ((n.id - u.id) > graph_size // 2):
                    continue

                open.add(u)
                nodes.add(u)
    return nodes


def is_valid_partitioning(graph: Graph, edge_weight_function):
    """
    check if we only send tensors between partitions
    """
    for n in graph.nodes:
        if n.value_type in {type(None), list, tuple, dict, set, int, bool, float, str, slice, torch.Size, torch.dtype}:
            for o in n.out_edges:
                if n.stage_id != o.stage_id:
                    msg = f"invalid output type at partition boundary {n.stage_id}=>{o.stage_id}"
                    msg += f"\noutput is {n.scope} of type {n.value_type}, weight {edge_weight_function(n, o)}"
                    return False, msg

    return True, ""


def print_all_problematic_outputs_between_partitions(graph: Graph, edge_weight_function):
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
                    msg += f"\noutput is {n.scope} of type {n.value_type}, weight {edge_weight_function(n, o)}"
                    valid_state = False
                    problems.append(msg)

    s = f"Valid outputs states = {valid_state}\n" + "problems:\n" + "\n".join(problems)
    print(s)
