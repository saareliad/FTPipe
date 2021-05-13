import math
import warnings
from typing import Optional, List, Tuple, Any

from tqdm import tqdm

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, \
    EdgeWeightFunction, NodeMemoryEstimator
from autopipe.autopipe.model_partitioning.mixed_pipe.post_process import post_process_partition
from autopipe.autopipe.model_profiling.control_flow_graph import Graph


# A must needed fix for PipeDream modeling of communication...
def calc_data_parall_comm_time_orig(num_machines, total_parameter_size, network_bandwidth):
    return ((4 * (num_machines - 1) * total_parameter_size) /
            (network_bandwidth * num_machines))


def calc_data_parall_comm_time_fixed(num_machines, total_parameter_size, network_bandwidth):
    # ring all-reduce is 2*D*(m-1)/m
    # afterwards they div by num machines, hence
    return 2 * ((4 * (num_machines - 1) * total_parameter_size) /
                (network_bandwidth))


calc_data_parall_comm_time = calc_data_parall_comm_time_fixed


def partition_pipedream(
        graph: Graph,
        num_gpus: int,
        node_weight_function: Optional[NodeWeightFunction] = None,
        edge_weight_function: Optional[EdgeWeightFunction] = None,
        use_layers_graph: bool = True,
        num_machines_in_first_level=None,
        memory_size = 10e9 ,#math.inf,
        verbose=True,
        force_stright_pipeline=True,
        node_mem_estimator: NodeMemoryEstimator = NodeMemoryEstimator(), # TODO: opt multiply

):
    print("PipeDream Partitioning")
    num_machines = num_gpus
    # we want results to be in milliseconds,
    network_bandwidth = edge_weight_function.bw * 1e6  # GBps -> KBps

    assert use_layers_graph
    graph.topo_sort()
    if use_layers_graph:
        work_graph, lookup = graph.layers_graph()
    else:
        work_graph, lookup = graph, None

    # saved_work_graph = work_graph
    # saved_work_graph_without_par_edges = saved_work_graph._remove_parallel_edges()  # creates a copy
    work_graph._remove_parallel_edges()

    # params_per_node = calculate_params_per_node(model, work_graph)

    non_input_nodes = list(work_graph.non_input_nodes)
    num_non_input_nodes = len(non_input_nodes)

    # Init A
    A: List[List[Tuple[Any, Any]]] = []
    for i in range(len(non_input_nodes)):
        row_A = []
        for j in range(num_machines):
            row_A.append((None, None))
        A.append(row_A)

    print("-I- Initializing data-parallel stage T_i_j")
    cum_sum = 0.0
    cum_activation_size = 0.0
    cum_parameter_size = 0.0
    something_works = False
    for i, node in enumerate(work_graph.non_input_nodes):
        cum_sum += node_weight_function(node)
        cum_activation_size += node_mem_estimator(node)  # FIXME: misleading name, it is aggregation
        cum_parameter_size += node.num_parameters
        # if force_stright_pipeline:
        max_m = 1 if force_stright_pipeline else num_machines
        for j in range(max_m):
            # stashed_data_size = math.ceil((num_machines - (j+1)) / (j+1)) * cum_activation_size
            # stashed_data_size += cum_parameter_size
            stashed_data_size = cum_activation_size
            if stashed_data_size > memory_size:
                A[i][j] = (None, None)
                # print("skipping OOM solution")
                continue

            data_parallel_communication_time = calc_data_parall_comm_time(num_machines=j + 1,
                                                                          total_parameter_size=cum_parameter_size,
                                                                          network_bandwidth=network_bandwidth)
            if num_machines_in_first_level is not None and j != (num_machines_in_first_level - 1):
                A[i][j] = (None, None)
            else:
                A[i][j] = (max(cum_sum, data_parallel_communication_time) / (j + 1), None)
                something_works = True
    print("-I- Done")
    if not something_works:
        warnings.warn("can't run any node without extra memory reduction - need to combine with other memory reduction methods")

    min_machines = 1 if num_machines_in_first_level is None else num_machines_in_first_level
    cum_times = []
    cum_activation_sizes = []
    cum_parameter_sizes = []
    for i, node in enumerate(work_graph.non_input_nodes):
        if i == 0:
            cum_times.append(node_weight_function(node))
            # cum_activation_sizes.append(profile_data[i][1])
            cum_activation_sizes.append(node_mem_estimator(node))
            cum_parameter_sizes.append(node.num_parameters)
        else:
            cum_times.append(cum_times[-1] + node_weight_function(node))
            # cum_activation_sizes.append(cum_activation_sizes[-1] + profile_data[i][1])
            cum_activation_sizes.append(cum_activation_sizes[-1] + node_mem_estimator(node))
            cum_parameter_sizes.append(cum_parameter_sizes[-1] + node.num_parameters)

    assert edge_weight_function.ratio == 0
    print("starting the optimization vs pipeline")
    for m in tqdm(range(min_machines, num_machines), desc="machines"):
        for i in tqdm(range(1, len(non_input_nodes)), desc="i"):
            node = non_input_nodes[i]
            (min_pipeline_time, optimal_split) = A[i][m]
            for j in range(i):
                max_m = 2 if force_stright_pipeline else m + 1
                for m_prime in range(1, max_m):
                    input_transfer_time = 2.0 * sum(edge_weight_function(nn, node) for nn in node.in_edges) / m_prime
                    output_transfer_time = 2.0 * sum(edge_weight_function(node, nn) for nn in node.out_edges) / m_prime

                    # outgoing_edges, outgoing_nodes, incoming_edges, incoming_nodes = cwf.calculate_borders([node])
                    # comm_bwd, comm_fwd = cwf.calculate_comm_forward_and_backward(incoming_edges, outgoing_edges)
                    # input_transfer_time = comm_bwd / m_prime
                    # output_transfer_time = comm_fwd / m_prime

                    # input_transfer_time = (2.0 * sum(nn node.in_edges) / (network_bandwidth * m_prime)
                    # input_transfer_time = (2.0 * profile_data[j][1]) / (network_bandwidth * m_prime)
                    # output_transfer_time = None
                    # if i < len(profile_data) -1:
                    #     output_transfer_time = (2.0 * profile_data[i-1][1]) / (network_bandwidth * m_prime)

                    last_stage_time = cum_times[i] - cum_times[j]
                    last_stage_parameter_size = cum_parameter_sizes[i] - cum_parameter_sizes[j]
                    last_stage_activation = cum_activation_sizes[i] - cum_activation_sizes[j]
                    stashed_data_size = last_stage_activation # param is included
                    # stashed_data_size = (cum_activation_sizes[i] - cum_activation_sizes[j])
                    # stashed_data_size *= math.ceil((num_machines - (m+1)) / m_prime)
                    # stashed_data_size += last_stage_parameter_size
                    if stashed_data_size > memory_size:
                        # print("skipping OOM solution")
                        continue

                    last_stage_time = max(last_stage_time,
                                          calc_data_parall_comm_time(num_machines=m_prime,
                                                                     total_parameter_size=last_stage_parameter_size,
                                                                     network_bandwidth=network_bandwidth))
                    last_stage_time /= m_prime

                    if A[j][m - m_prime][0] is None:
                        continue
                    pipeline_time = max(A[j][m - m_prime][0], last_stage_time, input_transfer_time)
                    if output_transfer_time is not None:
                        pipeline_time = max(pipeline_time, output_transfer_time)
                    if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                        optimal_split = (j, m - m_prime)
                        min_pipeline_time = pipeline_time
            A[i][m] = (min_pipeline_time, optimal_split)

    metadata = A[len(non_input_nodes) - 1][num_machines - 1]
    next_split = metadata[1]
    remaining_machines_left = num_machines
    splits = []
    replication_factors = []
    prev_split = len(non_input_nodes)
    while next_split is not None:
        num_machines_used = (remaining_machines_left - next_split[1] - 1)
        if verbose:
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
        splits.append(next_split[0] + 1)
        compute_time = 0.0
        parameter_size = 0.0
        for i in range(next_split[0] + 1, prev_split):
            node = non_input_nodes[i]
            compute_time += node_weight_function(node)
            parameter_size += node.num_parameters

        dp_communication_time = calc_data_parall_comm_time(num_machines=num_machines_used,
                                                           total_parameter_size=parameter_size,
                                                           network_bandwidth=network_bandwidth)

        # pp_communication_time_input = (profile_data[next_split[0]][1] * (1.0 / float(num_machines_used))) / network_bandwidth
        # pp_communication_time_output = (profile_data[prev_split-1][1] * (1.0 / float(num_machines_used))) / network_bandwidth

        node = non_input_nodes[next_split[0]]
        pp_communication_time_input = sum(edge_weight_function(nn, node) for nn in node.in_edges) / num_machines_used
        node = non_input_nodes[prev_split - 1]
        pp_communication_time_output = sum(edge_weight_function(node, nn) for nn in node.out_edges) / num_machines_used

        compute_time /= num_machines_used
        dp_communication_time /= num_machines_used
        if verbose:
            print(
                "Compute time = %f, Data-parallel communication time = %f, Pipeline-parallel communication time = %f..." % (
                    compute_time, dp_communication_time,
                    max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1]
        metadata = A[next_split[0]][next_split[1]]
        next_split = metadata[1]
        replication_factors.append(num_machines_used)
        remaining_machines_left -= num_machines_used
    if verbose:
        print("Number of machines used: %d..." % remaining_machines_left)
    num_machines_used = remaining_machines_left
    compute_time = 0.0
    parameter_size = 0.0
    for i in range(prev_split):
        node = non_input_nodes[i]
        compute_time += node_weight_function(non_input_nodes[i])
        parameter_size += node.num_parameters
    dp_communication_time = calc_data_parall_comm_time(num_machines=num_machines_used,
                                                       total_parameter_size=parameter_size,
                                                       network_bandwidth=network_bandwidth)

    compute_time /= num_machines_used
    dp_communication_time /= num_machines_used

    if verbose:
        print("Compute time = %f, Data-parallel communication time = %f..." % (compute_time, dp_communication_time))
        print()
        print("(Split start, split end) / time taken per stage / replication factor per stage:")

    prev_split = 0
    splits.reverse()
    splits.append(len(non_input_nodes))
    replication_factors.append(remaining_machines_left)
    replication_factors.reverse()

    for i in range(len(splits)):
        time = 0.0
        if verbose:
            print((prev_split, splits[i]), )
        for j in range(prev_split, splits[i]):
            time += node_weight_function(non_input_nodes[j])
        if verbose:
            print(time, replication_factors[i])
        prev_split = splits[i]

    total_time = 0.0
    total_parameter_size = 0.0
    for i in range(len(non_input_nodes)):
        node = non_input_nodes[i]
        total_time += node_weight_function(node)
        total_parameter_size += node.num_parameters

    data_parallel_communication_time = calc_data_parall_comm_time(num_machines=num_machines,
                                                                  total_parameter_size=total_parameter_size,
                                                                  network_bandwidth=network_bandwidth) / num_machines
    # data_parallel_total_time = max(total_time, data_parallel_communication_time) / num_machines
    data_parallel_total_time = (total_time / num_machines + data_parallel_communication_time)

    pipeline_parallel_total_time = A[len(non_input_nodes) - 1][num_machines - 1][0]

    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Throughput increase (compared to single machine):",
              total_time / pipeline_parallel_total_time)
        print("[Note that single-machine and %d-machine DP might not fit given memory constraints]")
        print("Throughput increase of %d-machine DP compared to single machine:" % num_machines,
              total_time / data_parallel_total_time)
        print("Throughput increase (compared to %d-machine DP):" % num_machines,
              data_parallel_total_time / pipeline_parallel_total_time)
        print("Number of images that need to be admitted:", int(math.ceil(
            float(num_machines) / replication_factors[0]) * replication_factors[0]))

        # Now, translate A to our graph.
        print("parameters", total_parameter_size)
        print("splits", splits)
        print("replication_factors", replication_factors)
        print("data_parallel_communication_time", data_parallel_communication_time)

    print(f"PipeDream returned {len(splits)} stages")

    # convert
    stage_id = 0
    start = 0
    for stop in splits:
        for n in non_input_nodes[start:stop]:
            n.stage_id = stage_id
        start = stop
        stage_id += 1

    # param per stage:
    start = 0
    params_per_stage = {i: 0 for i in range(len(splits))}
    for n in non_input_nodes:
        params_per_stage[n.stage_id] += n.num_parameters

    print("params per stage", params_per_stage)

    work_graph = post_process_partition(work_graph)

    # Note: this is a must
    # Copy work_graph -> saved_work_graph, since our graph is without parallel edges.
    if use_layers_graph:
        graph.induce_layer_partition(work_graph, lookup)

    if len(splits) != num_gpus:
        graph.serialize("saved_pipedream_non_stright_pipeline_graph")
        raise NotImplementedError("PipeDream returned non-straight pipeline")

    return graph


if __name__ == '__main__':
    from autopipe.autopipe.api import build_profiled_graph
    import torch
    from torch.nn import Sequential, Linear

    IN_FEATURES = 4 * 1024
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
    graph = build_profiled_graph(model, model_args=(inputs,), n_iter=10)

    MULT_FACTOR = 1
    node_weight_function = NodeWeightFunction(bwd_to_fwd_ratio=1, MULT_FACTOR=1)
    edge_weight_function = EdgeWeightFunction(bw_GBps=12, bwd_to_fwd_ratio=0, MULT_FACTOR=1, penalty=100000,
                                              ensure_positive=False)

    partition_pipedream(graph=graph, num_gpus=8,
                        node_weight_function=node_weight_function,
                        edge_weight_function=edge_weight_function,
                        use_layers_graph=True,
                        num_machines_in_first_level=None,
                        verbose=True)
