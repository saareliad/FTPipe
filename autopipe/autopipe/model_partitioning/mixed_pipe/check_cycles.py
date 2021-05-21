from typing import Set

import networkx as nx

from autopipe.autopipe.model_partitioning.heuristics import NodeMemoryEstimator
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node


def check_cycle2(g: Graph, a: Node, b: Node, nms=NodeMemoryEstimator()):
    """
    Checks if contracting (merging) (a,b) breaks topo order
    Args:
        g: topo-sorted graph
        a: start: first node in edge  (a,b)
        b: end: second node in edge (a,b)

    Returns:
        True if contracting would create a cycle.

    """
    # Add node AB, with outputs of combined A,B
    # start DFS from AB. if encountering A or B : there is a cycle.

    ab = Node(None, None, "dummy_not_None_scope")
    ab.out_edges = sorted(set(a.out_edges + b.out_edges) - {a, b}, key=lambda x: x.id)

    # TODO: dynamic topo sort. than we can just check path from a,b after removing edges.
    # https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/compiler/jit/graphcycles/graphcycles.cc#L19
    # https://cs.stackexchange.com/a/88325/130155
    # when dynamic topo sort is maintaned, we can
    # change depth_limit to rank b
    creates_a_cycle = g.forward_dfs_and_check_if_in_set(source=ab, set_to_check={a, b}, depth_limit=None)

    if not creates_a_cycle:
        if nms(a) + nms(b) > nms.THRESHOLD:
            return True  ### Failse due memory  # HACK #FIXME:

    return creates_a_cycle


def check_cycle_given_topo_sort(g: Graph, a: Node, b: Node):
    """
    # TODO: Requires topological order. (e.g dyanamic topo order)
    Checks if merging (a,b) breaks topo order
    Args:
        g: topo-sorted graph
        a: start: first node in edge  (a,b)
        b: end: second node in edge (a,b)
    Returns:
        True if there is another path (a->...->b) through missing nodes.
        (This means merging breaks topo order)
    """

    src_ids = a.topo_sort_id
    dst_ids = b.topo_sort_id
    A = {a}
    B = {b}

    missing_ids = set(range(src_ids + 1, dst_ids + 1))
    set_inputs = set(g.inputs)
    missing_nodes_in_work_graph = [g[i] for i in missing_ids if (i in g and g[i] not in set_inputs)]

    edge_nodes: Set[Node] = set(missing_nodes_in_work_graph)
    edges = []
    for a in A:
        for c in a.out_edges:
            if c in edge_nodes:
                edges.append((0, c.topo_sort_id + 2))

    for c in edge_nodes:
        for nc in c.out_edges:
            if nc in edge_nodes:
                edges.append((c.topo_sort_id + 2, nc.topo_sort_id + 2))
            elif nc in B:
                edges.append((c.topo_sort_id + 2, 1))

    G = nx.DiGraph(incoming_graph_data=edges)
    G.add_node(0)
    G.add_node(1)
    has_path = nx.algorithms.shortest_paths.generic.has_path(G, 0, 1)
    # Scream if has path
    is_ok = not has_path
    has_path_via_missing_nodes = not is_ok

    if not has_path_via_missing_nodes:
        for nn in b.in_edges:
            if nn.topo_sort_id > a.topo_sort_id:
                return True  # TODO: this doesn't mean a cycle!
    return has_path_via_missing_nodes