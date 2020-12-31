import warnings
from copy import deepcopy
from typing import List, Optional, Tuple

from sortedcollections import ValueSortedDict

from autopipe.autopipe.model_partitioning.heuristics import EdgeWeightFunction, NodeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_partitioning.mixed_pipe.systematic_block_ratio_creation import RatioBlockCreator
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node
from autopipe.autopipe.union_find import UnionFind


def coarsening(graph, edge_weight_function: EdgeWeightFunction, node_weight_function: NodeWeightFunction, L,
               matching_heuristics=List[str]) -> List[Tuple[Graph, List[List[Node]], Graph, UnionFind]]:
    # elements = set(i.id for i in itertools.chain.from_iterable(matching))
    uf = UnionFind(elements=[n.id for n in graph.non_input_nodes])

    print(f"-I- Coarsening: got graph with {graph.num_nodes} nodes")
    hierarchy = []

    p = graph
    matching = penalty_edges_matching(graph=p, edge_weight_function=edge_weight_function)
    g = contract(p, matching, edge_weight_function, uf=uf)
    hierarchy.append((p, matching, g, deepcopy(uf)))
    p = g

    # TODO: heavy edge matching on percentile

    p, _, g, uf, uf2  = online_heavy_edge_matching(p,
                                                  node_weight_function,
                                                  edge_weight_function,
                                                  L,
                                                  uf,
                                                  verbose=True,
                                                  record_history=True,
                                                  pecentile_to_filter=0.9)

    # it is the last so we dont deepcopy(uf)
    hierarchy.append((p, uf2, g, deepcopy(uf)))
    p = g


    # TODO: systematic-block-ratio-creation
    p, _, g, uf, uf2  = comm_comp_ratio_matching(
        p,
        node_weight_function,
        edge_weight_function,
        L,
        uf,
        verbose=True,
    )
    if uf2 is None:
        warnings.warn("can't restore single step of systematic max blocks")

    hierarchy.append((p, uf2, g, deepcopy(uf)))
    p = g


    p, _, g, uf, uf2 = online_smallest_comp_node_matching(p,
                                                          node_weight_function,
                                                          edge_weight_function,
                                                          L,
                                                          uf,
                                                          verbose=True,
                                                          record_history=True
                                                          )
    # it is the last so we dont deepcopy(uf)
    hierarchy.append((p, uf2, g, uf))
    p = g
    return hierarchy


def contract(graph: Graph, matching: List[List[Node]], edge_weight_function: EdgeWeightFunction,
             uf: Optional[UnionFind] = None) -> Graph:
    # if not matching:
    #     return graph
    new_graph = Graph.from_other(graph)
    # Start from end, so when we merge outputs are already handled
    for m in sorted(matching, key=lambda x: x[0].id, reverse=True):
        root = m[0]
        for i in m[1:]:
            new_graph.merge(root.id, i.id, edge_weight_function=edge_weight_function, uf=uf)
            if uf is not None:
                uf.union(x=root.id, y=i.id)
    return new_graph


def penalty_edges_matching(graph: Graph, edge_weight_function: EdgeWeightFunction):
    """Penalized edges are for disallowing sending weird stuff which MPI and the like can't handle.
        # TODO: if this creates a cycle we have nothing to do, but manually wrap it and disallow communication of weird stuff
    """
    matching = []
    for node in graph.non_input_nodes:
        check = False
        for out in node.out_edges:
            if edge_weight_function(node, out) == edge_weight_function.penalty:
                if check_cycle2(graph, node, out):
                    warnings.warn(f"can't compress edge with penalty (node,out)={(node, out)}")
                    continue
                matching.append([node, out])  # <---- into node
                # TODO: we have to handle doubles and so on...
                check = True
        if check:
            for e in matching[-len(node.out_edges):]:
                if e[0] != node:
                    # This should happen, since penalty is on the node itself
                    raise NotImplementedError(
                        f"potential cycle in edge {e}. Should probably duplicate node, or check topo order.")
    return matching


def code_analysis_matching(graph: Graph):
    pass


def adjacent_and_same_size_matching(graph: Graph):
    pass


def comm_comp_ratio_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                                       verbose=False):
    prev_graph = graph.from_other(graph)

    rbc = RatioBlockCreator(graph, edge_weight_function=edge_weight_function, node_weight_function=node_weight_function,
                            uf=uf)
    rbc.apply(L, verbose=verbose)

    matching = None
    return prev_graph, matching, graph, uf, None


def online_smallest_comp_node_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                                       verbose=False, record_history=False):
    # node_to_weight = dict(sorted(graph.non_input_nodes, key=lambda n: node_weight_function(n)))
    prev_graph = graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())

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
                uf2.union(u.id, v.id)
                hd.pop(u)
                hd.pop(v)
                hd[u] = node_weight_function(u)
                return True, weight_of_u
        return False, None

    history_sizes = []
    history_weights = []
    while len(hd) > L:
        # u, weight_of_u = hd.peekitem()
        merged_something, weight_of_u = inner_loop()
        if not merged_something:
            break
        if record_history:
            history_sizes.append(len(hd) + 1)
            history_weights.append(weight_of_u)
        if verbose:
            print(f"Nodes: {len(hd)}, Smallest: {weight_of_u}")

    # Note: matching is pretty much meaningless.
    matching = None
    return prev_graph, matching, graph, uf, uf2


def ofline_smallest_comp_node_matching(graph: Graph, node_weight_function):
    matching = []
    matched = set()

    for u in sorted(graph.non_input_nodes, key=lambda n: node_weight_function(n)):
        # Try to find match:
        if u in matched:
            continue
        for v in sorted(u.out_edges, key=lambda n: node_weight_function(n)):
            if v in matched:
                continue
            if check_cycle2(graph, u, v):
                # can't merge without breaking topo sort
                continue
            matched.add(u)
            matched.add(v)
            matching.append((u, v))
    return matching


def online_heavy_edge_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                               verbose=False, record_history=False, pecentile_to_filter=0.9):
    # node_to_weight = dict(sorted(graph.non_input_nodes, key=lambda n: node_weight_function(n)))
    prev_graph = graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())
    # HACK: re-using code from RatioBlockCreator
    rbc = RatioBlockCreator(graph, edge_weight_function=edge_weight_function, node_weight_function=node_weight_function,
                            uf=uf)

    hd = rbc.sorted_graph_forward_edges(descending=True)  # ValueSortedDict

    def inner_loop():
        # optimization: can use the index of new item to skip initial checks if there is no match in them.
        # But it works good enough without it.
        for (uid, vid), weight_of_u_v in hd.items():
            u = graph[uid]
            v = graph[vid]

            # Try to find match:
            if check_cycle2(graph, u, v):
                # can't merge without breaking topo sort
                continue
            graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
            # uf.union(u.id, v.id) handled below
            uf2.union(u.id, v.id)
            rbc.update_sorted_edges_on_merges(edges_to_value=hd, merges=[(u.id, v.id)], allow_poped_outside=True)
            return True, weight_of_u_v
        return False, None

    history_sizes = []
    history_weights = []
    import pandas as pd
    s = pd.Series(list(hd.values()))
    description = s.describe(percentiles=[0.5, 0.75, 0.8, 0.9, 0.95, 0.99])
    print(description)

    if pecentile_to_filter is not None:
        dest_length = len(hd) * pecentile_to_filter
        print(f"Filtering hte {pecentile_to_filter} percentile")
    else:
        dest_length = L

    while len(hd) > dest_length:
        # u, weight_of_u = hd.peekitem()
        merged_something, weight_of_merged = inner_loop()
        if not merged_something:
            break
        if record_history:
            history_sizes.append(len(hd) + 1)
            history_weights.append(weight_of_merged)
        if verbose:
            print(f"Edges: {len(hd)}, Largest edge: {weight_of_merged}")

    # Note: matching is pretty much meaningless.
    matching = None
    return prev_graph, matching, graph, uf, uf2
