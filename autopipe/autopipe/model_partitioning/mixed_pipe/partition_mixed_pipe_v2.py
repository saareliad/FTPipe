import multiprocessing
import warnings
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional, List, Tuple, Set

import networkx as nx
from sortedcollections import ValueSortedDict

from autopipe.autopipe.model_partitioning.heuristics import EdgeWeightFunction
from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.heap_dict import heapdict
from autopipe.autopipe.model_partitioning.mixed_pipe.partition_mixed_pipe import stages_from_bins, \
    convert_handle_missing_print
from autopipe.autopipe.model_partitioning.mixed_pipe.post_process import post_process_partition
from autopipe.autopipe.model_partitioning.mixed_pipe.refine import refine
from autopipe.autopipe.union_find import UnionFind
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node


def _lworker(args):
    (times, work_graph, best_objective) = lworker(*args)
    return times, work_graph.state(), best_objective


def lworker(L, P, edge_weight_function, node_weight_function, round_limit, saved_work_graph_without_par_edges):
    work_graph = Graph(None, None, None, None, None).load_state(saved_work_graph_without_par_edges)
    # work_graph = Graph.from_other(saved_work_graph_without_par_edges)
    # coarsening
    hierarchy = coarsening(work_graph, edge_weight_function, node_weight_function, L)
    # the output here should be L stages,
    last_graph: Graph = hierarchy[-1][-2]
    print(f"After coarsening: got best effort graph with {len(last_graph)} nodes (required: L={L})")
    # greedy load balance
    bins = greedy_best_fit(last_graph, P, node_weight_function)
    # TODO: print more stats, e.g comp/comm ratio, its interesting
    times = {i: sum(node_weight_function(x) for x in bins[i]) for i in bins}
    print("bin times:")
    pprint(times)
    # Bins to GPUs:
    for i, bin_nodes in bins.items():
        for n in bin_nodes:
            n.gpu_id = i
    # TODO: remove redundancy, its re-creating bins...
    last_graph.topo_sort(verbose=False, change_graph=False)
    id_to_node_worked_on = {n.topo_sort_id: n for n in last_graph.non_input_nodes}
    # Note: assert_missing_in_bins=False, since its the coarsened graph with many missing ids.
    # TODO: it has to be topo-sorted!
    n_stages = stages_from_bins(last_graph, bins, id_to_node_worked_on=id_to_node_worked_on,
                                assert_missing_in_bins=False, verbose=False)
    post_process_partition(last_graph, edge_weight_function=edge_weight_function, verbose_check_outputs=False)
    print(f"After greedy assignment: got {n_stages} stages")
    # un-coarsening
    first_graph = work_graph
    full_uf: UnionFind = hierarchy[-1][-1]
    # copy stage_ids and gpu_ids.
    component_mapping = full_uf.component_mapping()
    for topo_sort_id, node in id_to_node_worked_on.items():
        for i in component_mapping[node.id]:
            a = first_graph[i]
            b = last_graph[node.id]
            a.stage_id = b.stage_id
            a.gpu_id = b.gpu_id
    # Refinement
    best_objective = refine(work_graph, node_weight_function=node_weight_function, edge_weight_function=edge_weight_function,
           round_limit=round_limit)

    return times, work_graph, best_objective


def partition_mpipe(graph: Graph,
                    num_gpus: int,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_graph: bool = True,
                    round_limit=-1,
                    nprocs=1,
                    L_list=None,
                    **kwargs
                    ):
    print("mpipe got kwargs:", kwargs.keys())
    assert use_layers_graph
    graph.topo_sort()
    if use_layers_graph:
        work_graph, lookup = graph.layers_graph()
    else:
        work_graph, lookup = graph, None

    P = num_gpus
    # TODO: Choose L, loop over L's
    saved_work_graph = work_graph
    saved_work_graph_without_par_edges = saved_work_graph._remove_parallel_edges()  # creates a copy

    L_to_res = dict()
    # max_num = min(len(saved_work_graph) - len(list(saved_work_graph.inputs)) + 1, 3 * P + 1)
    # L_list = list(range(P, max_num))
    # L_list = list(range(2 * P, 2 * P + 1))  # FIXME: debugging cycles
    # L_list = list(range(8 * P, 8 * P + 1))  # FIXME: debugging cycles
    # L_list = [2*P, 4*P, 8*P, 16*P]
    if L_list is None:
        L_list=[P, 2*P, 3*P, 4*P, 5*P, 6*P, 7*P, 8*P]
        warnings.warn(f"no L_list given. using mine {L_list}")

    if nprocs > 1 and len(L_list) > 1:
        # Parallel version
        worker_args = [(L, P, edge_weight_function, node_weight_function, round_limit,
                        saved_work_graph_without_par_edges.state()) for L in L_list]

        with multiprocessing.Pool(min(nprocs, len(L_list))) as pool:
            results = pool.map(_lworker, worker_args)

        for L, (times, work_graph_state, best_objective) in zip(L_list, results):
            work_graph = Graph.from_state(work_graph_state)
            L_to_res[L] = (work_graph, times, best_objective)
    else:
        # sequential version
        for L in L_list:
            times, work_graph, best_objective = lworker(L, P, edge_weight_function, node_weight_function, round_limit,
                                        saved_work_graph_without_par_edges.state())

            # save res
            L_to_res[L] = (work_graph, times, best_objective)

    # TODO: choose best L times
    # TODO: fix the assert to torch.allclose, it fails due float round error even though same number.
    # x = [sum(times.values()) for L, (work_graph, times) in L_to_res.items()]
    # assert all([a == x[0] for a in x]), x

    # s2
    best_Ls2 = None
    minmax = None
    L_to_minmax = dict()
    # s3
    best_L = None
    best_objective_so_far = None

    best_objective_so_far = None
    L_to_best_objective = dict()

    L_to_num_stages = dict()


    for L, (work_graph, times, best_objective) in L_to_res.items():
        # s2
        worstcase = max(times.values())
        if best_Ls2 is None:
            minmax = worstcase
            best_Ls2 = L
        elif worstcase < minmax:
            best_Ls2 = L
            minmax = worstcase
        L_to_minmax[L] = worstcase

        # s3
        if best_L is None:
            best_objective_so_far = best_objective
            best_L = L
        elif best_objective_so_far > best_objective:
            best_L = L
            best_objective_so_far = best_objective

        L_to_best_objective[L] = best_objective

        # stages
        nstages = work_graph.num_partitions
        if None in work_graph.unique_partitions_ids:
            nstages -= 1
        L_to_num_stages[L] = nstages

    L = best_L
    work_graph = L_to_res[L][0]
    print(f"Best L is {L}")
    print("L_to_minmax (stage2):", L_to_minmax)
    print("L_to_num_stages:", L_to_num_stages)
    print("L_to_best_objective", L_to_best_objective)

    # # bins to stages
    # stages_from_bins(work_graph, bins, id_to_node_worked_on=id_to_node)

    work_graph = post_process_partition(work_graph)

    # Note: this is a must
    # Copy work_graph -> saved_work_graph, since our graph is without parallel edges.
    if use_layers_graph:
        graph.induce_layer_partition(work_graph, lookup)

    bins = defaultdict(list)
    for node in graph.nodes:
        if node.gpu_id is None:
            continue
        bins[node.gpu_id].append(node)

    stage_to_gpu_map = convert_handle_missing_print(bins, graph)

    return graph, stage_to_gpu_map


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


def check_cycle2(g: Graph, a: Node, b: Node):
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

    ab = Node(None, None, None)
    ab.out_edges = sorted(set(a.out_edges + b.out_edges) - {a, b}, key=lambda x: x.id)

    # TODO: dynamic topo sort. than we can just check path from a,b after removing edges.
    # https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/compiler/jit/graphcycles/graphcycles.cc#L19
    # https://cs.stackexchange.com/a/88325/130155
    # when dynamic topo sort is maintaned, we can
    # change depth_limit to rank b
    creates_a_cycle = g.forward_dfs_and_check(source=ab, set_to_check={a, b}, depth_limit=None)
    return creates_a_cycle


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


def main():
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


if __name__ == '__main__':
    main()
