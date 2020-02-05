from collections import defaultdict
from typing import List, Dict, Set

from ..model_profiling import Graph, NodeTypes

__all__ = ["post_process_partition"]


def post_process_partition(graph: Graph) -> Graph:
    '''
    process the partition and optimize it
    called as part of partition_graph method

    Parameters:
    ----------
    graph:
        the Graph object that was partitioned
    '''

    cannonize_partition_indices(graph)
    constants_fix(graph)
    remove_backward_edges(graph)
    do_not_send_lists(graph)
    constants_fix(graph)

    return graph


def cannonize_partition_indices(graph: Graph):
    out_edges = defaultdict(set)
    for node in graph.nodes:
        for o in node.out_nodes:
            out_edges[node.part].add(o.part)

    for i, e in out_edges.items():
        e.discard(i)

    translation = {idx: i for i, idx in enumerate(topological_sort(out_edges))}
    for node in graph.nodes:
        node.part = translation[node.part]


def _topological_sort(out_edges: Dict[int, Set[int]], v: int, visited: Dict[int, bool], stack: List[int]):
    visited[v] = True

    for i in out_edges[v]:
        if not visited[i]:
            _topological_sort(out_edges, i, visited, stack)

    stack.insert(0, v)


def topological_sort(out_edges: Dict[int, Set[int]]) -> List[int]:
    visited = {i: False for i in out_edges}
    stack = []

    for i in out_edges.keys():
        if not visited[i]:
            _topological_sort(out_edges, i, visited, stack)

    return stack


def constants_fix(graph: Graph):
    fixed = False
    while True:
        changed = False
        for node in graph.nodes:
            for n in node.in_nodes:
                if n.part != node.part and len(n.in_nodes) == 0 and n.type is NodeTypes.CONSTANT:
                    n.part = node.part
                    changed = True
                    fixed = True
        if not changed:
            break
    return fixed


def remove_backward_edges(graph: Graph):
    # TODO biased towards the lesser idx
    fixed = False
    while True:
        backward_edges = []
        for node in graph.nodes:
            for n in node.in_nodes:
                if n.part > node.part:
                    backward_edges.append((n, node))

        for u, v in backward_edges:
            v.part = u.part
            fixed = True

        if not backward_edges:
            break
    return fixed


def do_not_send_lists(graph: Graph):
    for node in graph.nodes:
        if not ('ListConstruct' in node.scope) or not ('TupleConstruct' in node.scope):
            continue

        # nodes that have a list input such as torch.cat
        # TODO revisit
        for out in node.out_nodes:
            if out.part != node.part:
                # fix the output
                out.part = node.part
                # fix the output other inputs
                for i in out.in_nodes:
                    i.part = node.part
