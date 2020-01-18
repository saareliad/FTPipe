from collections import deque
from typing import List

from ..model_profiling import Graph, NodeTypes

__all__ = ["post_process_partition"]


def post_process_partition(graph: Graph, part: List[int]) -> Graph:
    '''
    process the partition and optimize it
    called as part of partition_graph method

    Parameters:
    ----------
    graph:
        the Graph object that was partitioned
    part:
        a list of the nodes partition indices
    '''

    for node, idx in zip(graph.nodes, part):
        node.part = idx

    cannonize_partition_indices(graph)

    graph_root_fix(graph)
    remove_backward_edges(graph)
    do_not_send_lists(graph)

    return graph


def cannonize_partition_indices(graph: Graph):
    num_parts = len({n.part for n in graph.nodes})
    num_taken = 0
    model_inputs = [node for node in graph.nodes
                    if node.type == NodeTypes.IN]
    open_nodes = deque(model_inputs)
    closed = set()
    cannonical_parts = dict()

    while num_taken < num_parts:
        node = open_nodes.popleft()
        if node in closed or node in open_nodes:
            continue
        if node.part not in cannonical_parts:
            cannonical_parts[node.part] = num_taken
            num_taken += 1

        closed.add(node)
        edges = node.out_nodes.union(node.in_nodes)
        nodes = edges.difference(closed, set(open_nodes))
        open_nodes.extendleft(nodes)

    for node in graph.nodes:
        node.part = cannonical_parts[node.part]


def graph_root_fix(graph: Graph):
    fixed = False
    while True:
        changed = False
        for node in graph.nodes:
            for n in node.in_nodes:
                if n.part != node.part and len(n.in_nodes) == 0:
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
