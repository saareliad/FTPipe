import multiprocessing
import warnings
from collections import defaultdict
from pprint import pprint
from typing import Optional

from autopipe.autopipe.model_partitioning.heuristics import EdgeWeightFunction, NodeMemoryEstimator
from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.assignment import greedy_best_fit
from autopipe.autopipe.model_partitioning.mixed_pipe.coarsening import coarsening
from autopipe.autopipe.model_partitioning.mixed_pipe.partition_mixed_pipe import stages_from_bins, \
    convert_handle_missing_print
from autopipe.autopipe.model_partitioning.mixed_pipe.post_process import post_process_partition
from autopipe.autopipe.model_partitioning.mixed_pipe.refine import refine
from autopipe.autopipe.model_profiling.control_flow_graph import Graph
from autopipe.autopipe.union_find import UnionFind


def _lworker(args):
    (times, work_graph, best_objective) = lworker(*args)
    return times, work_graph.state(), best_objective


def lworker(model, L, P, edge_weight_function, node_weight_function, round_limit, saved_work_graph_without_par_edges,
            node_mem_estimator,
            basic_blocks, special_blocks, depth
            ):
    work_graph = Graph.from_state(saved_work_graph_without_par_edges)
    # work_graph = Graph.from_other(saved_work_graph_without_par_edges)
    # coarsening
    hierarchy = coarsening(model, work_graph, edge_weight_function, node_weight_function, L, P, basic_blocks, special_blocks, depth)
    # the output here should be L stages,
    last_graph: Graph = hierarchy[-1][-2]
    print(f"After coarsening: got best effort graph with {len(last_graph)} nodes (required: L={L})")
    # greedy load balance
    bins = greedy_best_fit(last_graph, P, node_weight_function, node_mem_estimator)
    # TODO: print more stats, e.g comp/comm ratio, its interesting
    times = {i: sum(node_weight_function(x) for x in bins[i]) for i in bins}
    print("bin times greedy:")
    pprint(times)

    # bins = exhustive_search(last_graph, P, node_weight_function, node_mem_estimator, L)
    # times = {i: sum(node_weight_function(x) for x in bins[i]) for i in bins}
    # print("bin times exhaustive:")
    # pprint(times)

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
    print(f"Got {n_stages} stages after initial assignment.")
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
    best_objective = refine(work_graph, node_weight_function=node_weight_function,
                            edge_weight_function=edge_weight_function,
                            round_limit=round_limit)

    return times, work_graph, best_objective


def partition_mpipe(model, graph: Graph,
                    num_gpus: int,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    node_mem_estimator: NodeMemoryEstimator = NodeMemoryEstimator(optimizer_multiply=3),
                    use_layers_graph: bool = True,
                    round_limit=-1,
                    nprocs=1,
                    L_list=None,
                    basic_blocks=(),
                    special_blocks=(),
                    depth=1000,
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
        # L_list=[P, 2*P, 3*P, 4*P, 5*P, 6*P, 7*P, 8*P]
        L_list = [2 * P]
        warnings.warn(f"no L_list given. using mine {L_list}")

    if nprocs > 1 and len(L_list) > 1:
        warnings.warn("experimental: parallel run on L.")
        # Parallel version
        worker_args = [(model, L, P, edge_weight_function, node_weight_function, round_limit,
                        saved_work_graph_without_par_edges.state(), node_mem_estimator, basic_blocks, special_blocks, depth) for L in L_list]

        with multiprocessing.Pool(min(nprocs, len(L_list))) as pool:
            results = pool.map(_lworker, worker_args)

        for L, (times, work_graph_state, best_objective) in zip(L_list, results):
            work_graph = Graph.from_state(work_graph_state)
            L_to_res[L] = (work_graph, times, best_objective)
    else:
        # sequential version
        for L in L_list:
            times, work_graph, best_objective = lworker(model, L, P, edge_weight_function, node_weight_function, round_limit,
                                                        saved_work_graph_without_par_edges.state(), node_mem_estimator,
                                                        basic_blocks, special_blocks, depth)

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
    # TODO: SAVE THIS. (also save the graphs and run full exp)
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


# Heavy edge matching: in communications.


def main():
    from autopipe.autopipe.api import build_profiled_graph
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
