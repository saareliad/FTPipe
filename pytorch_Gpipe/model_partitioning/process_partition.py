from ..model_profiling import Graph
from typing import List
from collections import deque, Counter

__all__ = ["post_process_partition"]


def post_process_partition(graph: Graph, part: List[int]):
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
    cannonize_partition_indices(graph, part)

    make_partitions_change_only_at_end_of_scope(graph)

    # make sure every scc in the graph is not splitted between different parts
    scc_partition_correction(graph)


def scc_partition_correction(graph: Graph):
    # create the scc graph
    vertices = [v.idx for v in graph.nodes]
    edges = {}
    for v in graph.nodes:
        idx_out_nodes = [h.idx for h in v.out_nodes]
        edges.update({v.idx: idx_out_nodes})

    for scc in strongly_connected_components_iterative(vertices, edges):
        # check if the scc is splitted between 2 parts or more
        scc_parts = []
        for v in scc:
            if graph.nodes[v].part not in scc_parts:
                scc_parts.append(graph.nodes[v].part)
            if len(scc_parts) >= 2:
                break
        # if he is splitted:
        if len(scc_parts) >= 2:
            output_part = -1
            # find out what part edges go to from this scc
            for v in scc:
                for out in graph.nodes[v].out_nodes:
                    if out.idx not in scc:
                        output_part = graph.nodes[out.idx].part
                        break
                if output_part != -1:
                    break
            # update the scc part to the part we found
            for v in scc:
                graph.nodes[v].part = output_part


def strongly_connected_components_iterative(vertices, edges):
    identified = set()
    stack = []
    index = {}
    boundaries = []

    for v in vertices:
        if v not in index:
            to_do = [('VISIT', v)]
            while to_do:
                operation_type, v = to_do.pop()
                if operation_type == 'VISIT':
                    index[v] = len(stack)
                    stack.append(v)
                    boundaries.append(index[v])
                    to_do.append(('POSTVISIT', v))
                    # We reverse to keep the search order identical to that of
                    # the recursive code;  the reversal is not necessary for
                    # correctness, and can be omitted.
                    to_do.extend(
                        reversed([('VISITEDGE', w) for w in edges[v]]))
                elif operation_type == 'VISITEDGE':
                    if v not in index:
                        to_do.append(('VISIT', v))
                    elif v not in identified:
                        while index[v] < boundaries[-1]:
                            boundaries.pop()
                else:
                    # operation_type == 'POSTVISIT'
                    if boundaries[-1] == index[v]:
                        boundaries.pop()
                        scc = set(stack[index[v]:])
                        del stack[index[v]:]
                        identified.update(scc)
                        yield scc


def cannonize_partition_indices(graph: Graph, node_parts: List[int]):
    for node, part in zip(graph.nodes, node_parts):
        node.part = part

    num_parts = len(set(node_parts))
    num_taken = 0
    model_inputs = graph.nodes[graph.num_inputs:]
    open_nodes = deque([(n, 0)for n in model_inputs])
    closed = set()
    cannonical_parts = dict()

    while num_taken < num_parts:
        node, d = open_nodes.popleft()
        if node.part not in cannonical_parts:
            cannonical_parts[node.part] = num_taken
            num_taken += 1

        closed.add(node)
        edges = node.out_nodes.union(node.in_nodes)
        nodes = edges.difference(closed, set(open_nodes))
        open_nodes.extend([(n, d) for n in nodes])

    for node in graph.nodes:
        node.part = cannonical_parts[node.part]


def make_partitions_change_only_at_end_of_scope(graph: Graph):
    def is_first_in_partition(node):
        return any(other.part != node.part for other in node.in_nodes)

    first_nodes_of_partition = filter(is_first_in_partition, graph.nodes)

    for node in first_nodes_of_partition:
        scope_depth = node.scope.count('/')-1
        # dont do it too shallow
        if scope_depth >= 2:  # TODO think about threshold
            parent_scope = node.scope.rsplit('/', 1)[0]

            def in_scope(n):
                return parent_scope == n.scope.rsplit('/', 1)[0]
            scope_nodes = list(filter(in_scope, graph.nodes))
            parts = [n.part for n in scope_nodes]
            part_histogram = Counter(parts)
            most_common, num_layers = part_histogram.most_common(1)[0]
            if num_layers >= len(parts)//2:
                for other in scope_nodes:
                    other.part = most_common
