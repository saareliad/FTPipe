from copy import deepcopy
from typing import List, Set

import numpy as np
from sortedcollections import ValueSortedDict, SortedDict

from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node
from autopipe.autopipe.union_find import UnionFind
import torch


def full_alg(graph, P, L, node_weight_function, edge_weight_function, uf, rtol=2e-3):
    history = get_rep_analysis_history_with_online_smallest_comp_node_matching(graph,
                                                                               node_weight_function=node_weight_function,
                                                                               edge_weight_function=edge_weight_function,
                                                                               L=L,
                                                                               uf=uf,
                                                                               rtol=rtol,
                                                                               verbose=False)

    torch.save(history, "history.tmp")
    print("Saved history to file history.tmp")
    repetitive_adjacent_analysis(history, L, P)


def repetitive_adjacent_analysis(history: List[List[Set[Node]]], L, P):
    # TODO:
    for i, found_sets in enumerate(history):
        lengths = [len(x) for x in found_sets]
        print(f"-I- merge {i} Found set lengths {lengths}")
        # TODO: have to think about this

        for l in lengths:
            if l % P == 0:
                k = l // P
                print(f"    Found set of size {l} splitting it to {k}*{P} groups")
                # TODO: split to sets of K
                # TODO: there should be ability to undo the splitting to increase k. (or to do a dry run, 1st and 2nd passes)
            elif l > P:
                print(f"    Found set of size {l}, currently ignoring")
                # TODO: should track if the set size increases of decreases,
                # when ts starts decreasing: consider doing the merge.


def record_repetitive_adjacent(graph, node_weight_function, rtol=2e-3, do_topo_sort=True):
    if do_topo_sort:
        graph.topo_sort(change_graph=False)
    topo_sorted_nodes_to_weight = SortedDict({n.topo_sort_id: node_weight_function(n) for n in graph.non_input_nodes})

    found_sets = []
    cur = None
    rsum = 0
    cur_set = set()
    for node, weight in topo_sorted_nodes_to_weight.items():
        if cur is None:
            cur = weight
            cur_set.add(node)
            rsum = weight
        elif np.allclose(weight, cur, rtol):
            rsum += weight
            cur_set.add(node)
            cur = rsum / len(cur_set)
        else:
            if cur_set:
                # check how am I so far
                found_sets.append(cur_set)

            # clear search,
            cur = weight
            rsum = weight
            cur_set = set()

    return found_sets


def get_rep_analysis_history_with_online_smallest_comp_node_matching(graph: Graph, node_weight_function,
                                                                     edge_weight_function, L, uf: UnionFind,
                                                                     verbose=True, rtol=2e-3):
    prev_graph = Graph.from_other(graph)
    # switch: work on the dummy version
    graph, prev_graph = prev_graph, graph
    uf = deepcopy(uf)
    # Used to find the local multi-matching
    # uf2 = UnionFind(elements=graph._nodes.keys())

    hd = ValueSortedDict({
        n: node_weight_function(n) for n in graph.non_input_nodes
    })

    def inner_loop():
        # optimization: can use the index of new item to skip initial checks if there is no match in them.
        # But it works good enough without it.
        for u, weight_of_u in hd.items():
            # Try to find match:
            for v in sorted(u.out_edges, key=lambda n: node_weight_function(n)):
                if check_cycle2(graph, u, v):
                    # can't merge without breaking topo sort
                    continue
                graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
                uf.union(u.id, v.id)
                # uf2.union(u.id, v.id)
                hd.pop(u)
                hd.pop(v)
                hd[u] = node_weight_function(u)
                return True, weight_of_u
        return False, None

    rep_analysis_history = []
    while len(hd) > L:
        # u, weight_of_u = hd.peekitem()
        merged_something, weight_of_u = inner_loop()
        if not merged_something:
            break
        found_sets = record_repetitive_adjacent(graph, node_weight_function, rtol=rtol, do_topo_sort=True)
        rep_analysis_history.append(found_sets)
        if verbose:
            print(f"Nodes: {len(hd)}, Smallest: {weight_of_u}")

    return rep_analysis_history


if __name__ == '__main__':
    pass
