from collections import defaultdict
from typing import List, Dict, Set
from copy import copy
from ..model_profiling import Graph, NodeTypes, Node
import networkx as nx

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
    if has_cycles(graph):
        graph.save_as_pdf(f"{graph.model_name}_before_fix",
                          ".", show_profiles=True)
        break_partition_cycles(graph)

        # possibly redundent
        cannonize_partition_indices(graph)

    # this is a sanity check
    if has_cycles(graph):
        graph.save_as_pdf(f"{graph.model_name}_after_fix",
                          ".", show_profiles=True)
        error = "error cycle detected mutual dependecy between partitions"
        raise AssertionError(error)

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


def has_cycles(graph: Graph) -> bool:
    for u in graph.nodes:
        for v in u.out_nodes:
            if v.part < u.part:
                return True

    return False


def check_if_cycle_is_solvable(graph: Graph) -> bool:
    """ 
    check if a cycle detected is a solvable or not.
    If on the same partition we have:
    w->u and w->v but no "direct" connction u->v
    that is, the cycle is
    w->u->->x->v
    s.t
    w.part == u.part == v.part
    and x.part < u.part

    we can solve the cycle,
    using several solutions.

    For example:
        we can split
        w, u: new partition x.part - 1, (before x)
        v: new partition after x.part

        put both partitions on same GPU and share w.
        (So with cuda aware there is no communication sending w to v)
    """
    ngx = graph.asNetworkx(directed=True)
    has_true_cycle = True
    try:
        nx.find_cycle(ngx)
    except nx.NetworkXNoCycle:
        has_true_cycle = False
    return has_true_cycle


def break_partition_cycles(graph: Graph):
    parts = set()
    roots = defaultdict(set)
    # roots[i] = nodes in partition j s.t j<i and exists backward edge from partition i to j
    for u in graph.nodes:
        parts.add(u.part)
        for v in u.out_nodes:
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
        for u in n.out_nodes:
            if u.part == n.part:
                nodes.add(u)
                open.add(u)

    open = copy(nodes)
    while len(open) > 0:
        n = open.pop()
        if n in roots:
            continue

        for u in n.in_nodes:
            if u.part == n.part:
                # TODO we need to know if u is part of the sub tree
                # this is an reasonable estimation
                if u.type != NodeTypes.IN and ((n.idx - u.idx) > graph_size // 2):
                    continue

                open.add(u)
                nodes.add(u)
    return nodes
