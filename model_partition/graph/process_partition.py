import torch.nn as nn
import torch
from collections import deque
from .control_flow_graph import Graph, NodeTypes, Node

# TODO there was a case where the partition was not not connected densenet depth 100


def post_process_partition(graph: Graph, nparts, part, weights=None):
    set_partition(graph, part)
    # make sure the inputs to an OP type node are all in the same part
    OP_inputs_partition_correction(graph, nparts)
    # make sure every scc in the graph is not splitted between different parts
    scc_partition_correction(graph)
    # _check_bi_directional(graph, nparts)


# TODO fix
def OP_inputs_partition_correction(graph: Graph, nparts):
    node_to_best_part = dict()
    for v in graph.nodes:
        if v.type == NodeTypes.OP:
            # pick the part of the inputs as the one with least comunication
            group = {u for u in v.in_nodes}
            group.add(v)
            min_comunication = float('inf')
            best_part = -1
            for part in range(nparts):
                parts = []
                for u in group:
                    parts.append(graph.nodes[u.idx].part)
                    graph.nodes[u.idx].part = part

                comunication = compute_comunication(graph)
                if comunication < min_comunication:
                    min_comunication = comunication
                    best_part = part

            for u, p in zip(group, parts):
                node_to_best_part[u.idx] = best_part
                graph.nodes[u.idx].part = p

    for idx, best in node_to_best_part.items():
        graph.nodes[idx].part = best


def compute_comunication(graph: Graph):
    count = 0
    for v in graph.nodes:
        for u in v.in_nodes:
            if u.part != v.part:
                count += 1
    return count


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


def set_partition(graph: Graph, parts):
    for node, part in zip(graph.nodes, parts):
        node.part = part


def _check_bi_directional(graph, nparts):
    outputs_per_partition = map(
        _partition_out_nodes, _group_by_partition(graph, nparts))
    inputs = _partition_input_nodes(graph.nodes)

    part_to_device = _partition_to_device(list(range(nparts)), inputs)

    for node in graph.nodes:
        node.part = part_to_device[node.part]

    def out_parts(partition_outs):
        res = set(*map(lambda node: map(lambda n: n.part,
                                        node.out_nodes), partition_outs))
        res.discard(partition_outs[0].part)
        return res


# return a list where each element is a list of nodes belonging to the same partition
def _group_by_partition(graph: Graph, nparts):
    lst = [[] for _ in range(nparts)]
    for node in graph.nodes:
        lst[node.part].append(node)
    return lst


# return list of all nodes who are inputs of a partition
def _partition_input_nodes(nodes):
    def is_input_node(node):
        return not node.in_nodes or any(in_node.part != node.part for in_node in node.in_nodes)

    return list(filter(is_input_node, nodes))


def _partition_out_nodes(nodes):
    def is_output_node(node):
        return not node.out_nodes or any(out_node.part != node.part for out_node in node.out_nodes)

    return list(filter(is_output_node, nodes))


# map a partition index to a another set of indices
def _partition_to_device(normalized_ids, model_inputs):
    part_to_device = dict()
    num_taken = 0
    open_nodes = deque(model_inputs)
    closed = set()
    seen_parts = set()

    while num_taken < len(normalized_ids):
        # TODO there was an instance where it crushed here assuming there were less partitions then assumed
        node = open_nodes.popleft()
        if node.part not in seen_parts:
            part_to_device[node.part] = normalized_ids[num_taken]
            num_taken += 1
            seen_parts.add(node.part)

        closed.add(node)
        edges = node.out_nodes.union(node.in_nodes)

        open_nodes.extend(edges.difference(closed, set(open_nodes)))

    return part_to_device
