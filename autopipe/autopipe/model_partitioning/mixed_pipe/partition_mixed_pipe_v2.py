from copy import deepcopy
from pprint import pprint
from typing import Optional, List, Tuple, Set
from sortedcollections import ValueSortedDict

import networkx as nx

from autopipe.autopipe.model_partitioning.heuristics import EdgeWeightFunction
from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.heap_dict import heapdict
from autopipe.autopipe.model_partitioning.mixed_pipe.partition_mixed_pipe import stages_from_bins, \
    convert_handle_missing_print
from autopipe.autopipe.model_partitioning.mixed_pipe.post_process import post_process_partition
from autopipe.autopipe.model_partitioning.mixed_pipe.refine import refine
from autopipe.autopipe.model_partitioning.mixed_pipe.union_find import UnionFind
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node


def partition_mpipe(graph: Graph,
                    num_gpus: int,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_graph: bool = True,
                    round_limit=-1,
                    **kwargs
                    ):
    print("mpipe got kwargs:", kwargs.keys())
    graph.topo_sort()
    if use_layers_graph:
        work_graph, lookup = graph.layers_graph()
    else:
        work_graph, lookup = graph, None

    P = num_gpus
    # TODO: Choose L, loop over L's
    saved_work_graph = work_graph

    L_to_res = dict()
    max_num = min(len(saved_work_graph) - len(list(saved_work_graph.inputs)) + 1, 4*P+1)
    for L in range(P, max_num):
        work_graph = Graph.from_other(saved_work_graph)

        # TODO: coarsening
        hierarchy = coarsening(work_graph, edge_weight_function, node_weight_function, L)
        # TODO: the output here should be L stages,
        last_graph: Graph = hierarchy[-1][-2]
        print(f"After coarsening: got best effort graph with {len(last_graph)} nodes (required: L={L})")
        # TODO: greedy load balance
        bins = greedy_best_fit(last_graph, P, node_weight_function)

        # TODO: print more stats, e.g comp/comm ratio, its interesting
        times = {i: sum(node_weight_function(x) for x in bins[i]) for i in bins}
        print("bin times:")
        pprint(times)

        # Bins to GPUs:
        for i, bin_nodes in bins.items():
            for n in bin_nodes:
                n.gpu_id = i

        # TODO: remove redudency, its re-creating bins...
        id_to_node_worked_on = {n.id: n for n in last_graph.non_input_nodes}
        n_stages = stages_from_bins(last_graph, bins, id_to_node_worked_on=id_to_node_worked_on)
        print(f"After greedy assignment: got {n_stages} stages")
        # TODO: un-coarsening
        first_graph = work_graph
        full_uf: UnionFind = hierarchy[-1][-1]
        # copy stage_ids and gpu_ids.

        component_mapping = full_uf.component_mapping()
        for v in id_to_node_worked_on:
            for i in component_mapping[v]:
                a = first_graph[i]
                b = last_graph[v]
                a.stage_id = b.stage_id
                a.gpu_id = b.gpu_id
        # TODO Refinement
        refine(work_graph, node_weight_function=node_weight_function, edge_weight_function=edge_weight_function,
               round_limit=round_limit)

        # TODO: save res
        L_to_res[L] = (work_graph, times)

    # TODO: choose best L times
    # TODO: fix the assert to torch.allclose, it fails due float round error even though same number.
    # x = [sum(times.values()) for L, (work_graph, times) in L_to_res.items()]
    # assert all([a == x[0] for a in x]), x

    best_L = None
    minmax = None
    L_to_minmax = dict()
    for L, (work_graph, times) in L_to_res.items():
        worstcase = max(times.values())
        if best_L is None:
            minmax = worstcase
            best_L = L
        elif worstcase < minmax:
            best_L = L
            minmax = worstcase
        L_to_minmax[L] = worstcase

    L_to_num_stages = dict()
    for L, (work_graph, times) in L_to_res.items():
        nstages = work_graph.num_partitions
        if None in work_graph.unique_partitions_ids:
            nstages -=1
        L_to_num_stages[L] = nstages

    L = best_L
    work_graph = L_to_res[L][0]
    print(f"Best L is {L}")
    print("L_to_minmax:", L_to_minmax)
    print("L_to_num_stages:", L_to_num_stages)

    # # bins to stages
    # stages_from_bins(work_graph, bins, id_to_node_worked_on=id_to_node)

    work_graph = post_process_partition(work_graph)

    if use_layers_graph:
        graph.induce_layer_partition(work_graph, lookup)

    stage_to_gpu_map = convert_handle_missing_print(bins, graph)

    return graph, stage_to_gpu_map


def contract(graph: Graph, matching: List[List[Node]], uf: Optional[UnionFind] = None) -> Graph:
    # if not matching:
    #     return graph
    new_graph = Graph.from_other(graph)
    # Start from end, so when we merge outputs are already handled
    for m in sorted(matching, key=lambda x: x[0].id, reverse=True):
        root = m[0]
        for i in m[1:]:
            new_graph.merge(root.id, i.id)
            if uf is not None:
                uf.union(x=root.id, y=i.id)
    return new_graph


def coarsening(graph, edge_weight_function: EdgeWeightFunction, node_weight_function: NodeWeightFunction, L,
               matching_heuristics=List[str]) -> List[Tuple[Graph, List[List[Node]], Graph, UnionFind]]:
    # elements = set(i.id for i in itertools.chain.from_iterable(matching))
    uf = UnionFind(elements=[n.id for n in graph.non_input_nodes])

    num_nodes = graph.num_nodes
    hierarchy = []

    p = graph
    matching = penalty_edges_matching(graph=p, edge_weight_function=edge_weight_function)
    g = contract(p, matching, uf=uf)
    hierarchy.append((p, matching, g, deepcopy(uf)))
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


def check_cycle(g: Graph, a: Node, b: Node):
    """
    Checks if merging (a,b) breaks topo order
    Args:
        g: topo-sorted graph
        a: start: first node in edge  (a,b)
        b: end: second node in edge (a,b)
    Returns:
        True if there is another path (a->...->b) through missing nodes.
        (This means merging breaks topo order)
    """

    src_ids = a.id
    dst_ids = b.id
    A = {a}
    B = {b}

    missing_ids = set(range(src_ids + 1, dst_ids + 1))
    missing_nodes_in_work_graph = [g[i] for i in missing_ids if (i in g and g[i] not in g.inputs)]

    edge_nodes: Set[Node] = set(missing_nodes_in_work_graph)
    edges = []
    for a in A:
        for c in a.out_edges:
            if c in edge_nodes:
                edges.append((0, c.id + 2))

    for c in edge_nodes:
        for nc in c.out_edges:
            if nc in edge_nodes:
                edges.append((c.id + 2, nc.id + 2))
            elif nc in B:
                edges.append((c.id + 2, 1))

    G = nx.DiGraph(incoming_graph_data=edges)
    G.add_node(0)
    G.add_node(1)
    has_path = nx.algorithms.shortest_paths.generic.has_path(G, 0, 1)
    # Scream if has path
    is_ok = not has_path
    has_path_via_missing_nodes = not is_ok
    return has_path_via_missing_nodes


def penalty_edges_matching(graph: Graph, edge_weight_function: EdgeWeightFunction):
    """Penalized edges are for disallowing sending weird stuff which MPI and the like can't handle.
        # TODO: if this creates a cycle we have nothing to do, but manually wrap it and disallow communication of weird stuff
    """
    matching = []
    for node in graph.non_input_nodes:
        check = False
        for out in node.out_edges:
            if edge_weight_function(node, out) == edge_weight_function.penalty:
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


def comm_comp_ratio_matching(graph: Graph):
    pass


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
                if check_cycle(graph, u, v):
                    # can't merge without breaking topo sort
                    continue
                uf.union(u.id, v.id)
                uf2.union(u.id, v.id)
                hd.pop(u)
                hd.pop(v)
                graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function)
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
            if check_cycle(graph, u, v):
                # can't merge without breaking topo sort
                continue
            matched.add(u)
            matched.add(v)
            matching.append((u, v))
    return matching


def below_threshold_matching(graph: Graph):
    pass


# Heavy edge matching: in communications.

def greedy_best_fit(graph: Graph, P, node_weight_function, ):
    bins = {i: list() for i in range(P)}
    bin_weights = heapdict({i: 0 for i in range(P)})
    bin_memory = heapdict({i: 0 for i in range(P)})

    node_to_weight = {n: node_weight_function(n) for n in graph.non_input_nodes}
    node_to_weight = dict(sorted(node_to_weight.items(), key=lambda item: item[1], reverse=True))

    def check_memory_fit(candidate, bin_id):
        # TODO: implement after PoC
        return True

    def choose_bin(node):
        while bin_weights:
            bin_id, w = bin_weights.peekitem()
            if not check_memory_fit(node, bin_id):
                bin_weights.popitem()
                continue
            return bin_id
        raise RuntimeError("Could not find an assignment which fits memory")

    while node_to_weight:
        node, node_weight = node_to_weight.popitem()
        bin_id = choose_bin(node)
        bins[bin_id].append(node)
        bin_weights[bin_id] += node_weight
        # TODO: update memory
        bin_memory[bin_id] += 0

    return bins


if __name__ == '__main__':
    from autopipe.autopipe import build_profiled_graph
    import torch
    from torch.nn import Sequential, Linear

    IN_FEATURES = 320
    OUT_FEATURES = 8
    n_encoder_decoder = 12

    l = []
    for i in range(n_encoder_decoder):
        l.append(Linear(IN_FEATURES, IN_FEATURES))
    l.append(Linear(IN_FEATURES, OUT_FEATURES))
    for i in range(n_encoder_decoder):
        l.append(Linear(OUT_FEATURES, OUT_FEATURES))

    model = Sequential(*l)

    inputs = torch.randn(IN_FEATURES, IN_FEATURES)

    model = model.cuda()
    inputs = inputs.cuda()
    graph = build_profiled_graph(model, model_args=(inputs,), n_iter=50)

    node_weight_function = NodeWeightFunction(bwd_to_fwd_ratio=1, MULT_FACTOR=100000)
    edge_weight_function = EdgeWeightFunction(bw_GBps=12, bwd_to_fwd_ratio=0, MULT_FACTOR=100000, penalty=100000)
    # graph.display(node_weight_function=node_weight_function)
    # dot = graph.build_dot(node_weight_function=node_weight_function)
    # graphviz.Source(graph.build_dot(node_weight_function=node_weight_function))
    # nxg = graph.asNetworkx(directed=False, node_weight_function=node_weight_function)
    # import matplotlib.pyplot as plt
    # nx.draw_networkx(nxg, labels={n: {"weight": v["weight"]} for n,v in nxg.nodes.items()})
    # plt.show()

    # analyze_n_clusters(nodes=nodes, node_weight_function=node_weight_function, max_k=4)
    graph, stage_to_gpu_map = partition_mpipe(graph=graph, num_gpus=2,
                                              node_weight_function=node_weight_function,
                                              edge_weight_function=edge_weight_function,
                                              use_layers_graph=True)
