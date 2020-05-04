from collections import defaultdict
from typing import List, Dict, Set
from copy import copy
import os
from ..model_profiling import Graph, NodeTypes, Node


__all__ = ["post_process_partition"]


def post_process_partition(graph: Graph, verbose_on_error=True) -> Graph:
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

    cannonize_partition_indices(graph)
    if has_cycles(graph):
        if os.environ.get("DEBUG", False):
            graph.save_as_pdf(f"{graph.model_name}_before_fix",
                              ".", show_profiles=True)
        break_partition_cycles(graph)

        # possibly redundent
        try:
            cannonize_partition_indices(graph)
        except:
            print(
                "-W- ignorring exception of redundent cannonize_partition_indices(graph)")

    # this is a sanity check
    if has_cycles(graph):
        if os.environ.get("DEBUG", False):
            graph.save_as_pdf(f"{graph.model_name}_after_fix",
                              ".", show_profiles=True)

        if verbose_on_error:
            problems, info = get_problematic_partitions(graph)
            print("-V- printing problematic partitions")
            for p, i in zip(problems, info):
                print(p)
                print(i)

            n_partitions = len(set(u.part for u in graph.nodes))
            print("n_partitions:", n_partitions)

        error = "error cycle detected mutual dependecy between partitions"
        raise AssertionError(error)

    is_valid, error = is_valid_partitioning(graph)
    # TODO disabled until we have smart partitioning
    # assert is_valid, error

    return graph


def cannonize_partition_indices(graph: Graph):
    out_edges = defaultdict(set)
    for node in graph.nodes:
        for o in node.out_edges:
            out_edges[node.part].add(o.part)

    for i, e in out_edges.items():
        e.discard(i)

    translation = {idx: i for i, idx in enumerate(topological_sort(out_edges))}
    for node in graph.nodes:
        node.part = translation[node.part]


def topological_sort(out_edges: Dict[int, Set[int]]) -> List[int]:
    visited = {i: False for i in out_edges}
    stack = []

    for i in out_edges.keys():
        if not visited[i]:
            _topological_sort(out_edges, i, visited, stack)

    return stack


def _topological_sort(out_edges: Dict[int, Set[int]], v: int, visited: Dict[int, bool], stack: List[int]):
    visited[v] = True

    for i in out_edges[v]:
        if not visited[i]:
            _topological_sort(out_edges, i, visited, stack)

    stack.insert(0, v)


def get_problematic_partitions(graph):
    """ For debug when cycle are detected """
    problems = []
    info = []
    for u in graph.nodes:
        for v in u.out_edges:
            if v.part < u.part:
                problems.append([v.part, u.part])
                info.append([v, u])
    return problems, info


def has_cycles(graph: Graph) -> bool:
    for u in graph.nodes:
        for v in u.out_edges:
            if v.part < u.part:
                return True

    return False


def break_partition_cycles(graph: Graph):
    parts = set()
    roots = defaultdict(set)
    # roots[i] = nodes in partition j s.t j<i and exists backward edge from partition i to j
    for u in graph.nodes:
        parts.add(u.part)
        for v in u.out_edges:
            if u.part > v.part:
                roots[v.part].add(v)

    n_parts = len(parts)
    for idx, group in roots.items():
        # each group represents a new partition to create
        for n in find_subtree(group, len(graph.nodes)):
            n.part = n_parts
        n_parts += 1


def find_subtree(roots: Set[Node], graph_size: int):
    nodes = set()
    open = copy(roots)
    while len(open) > 0:
        n = open.pop()
        nodes.add(n)
        for u in n.out_edges:
            if u.part == n.part:
                nodes.add(u)
                open.add(u)

    open = copy(nodes)
    while len(open) > 0:
        n = open.pop()
        if n in roots:
            continue

        for u in n.in_edges:
            if u.part == n.part:
                # TODO we need to know if u is part of the sub tree
                # this is an reasonable estimation
                if u.type != NodeTypes.IN and ((n.id - u.id) > graph_size // 2):
                    continue

                open.add(u)
                nodes.add(u)
    return nodes


def is_valid_partitioning(graph: Graph):
    """
    check if we only send tensors between partitions
    """
    for n in graph.nodes:
        if n.value_type in {type(None), list, tuple, dict, set, int, bool, float, str, slice}:
            for o in n.out_edges:
                if n.part != o.part:
                    msg = f"invalid output type at partition boundary {n.part}=>{o.part}"
                    msg += f"\noutput is {n.scope} of type {n.value_type()}"
                    return False, msg

    return True, ""
