from itertools import combinations
from typing import Optional, List, Tuple, Any

from autopipe.autopipe.model_partitioning.acyclic.acyclic_partitioning import calculate_params_per_node
from autopipe.autopipe.model_partitioning.heuristics import CoarsenedWeightFunction, NodeWeightFunction, \
    EdgeWeightFunction
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node


def partition_pipedream(
        model,
        graph: Graph,
        num_gpus: int,
        node_weight_function: Optional[NodeWeightFunction] = None,
        edge_weight_function: Optional[EdgeWeightFunction] = None,
        use_layers_graph: bool = True,
        use_data_parallel=False,  # straight_pipeline
        num_machines=1,
        num_gpus_within_machine=None,
):
    cwf = CoarsenedWeightFunction(edge_weight_function=edge_weight_function,
                                  node_weight_function=node_weight_function,
                                  do_longest_path=False)
    assert use_layers_graph
    graph.topo_sort()
    if use_layers_graph:
        work_graph, lookup = graph.layers_graph()
    else:
        work_graph, lookup = graph, None

    # params_per_node = calculate_params_per_node(model, work_graph)

    if num_machines == 1:
        num_gpus_within_machine = num_gpus
    elif num_machines > 1:
        raise NotImplementedError()
    else:
        raise ValueError()

    non_input_nodes = list(work_graph.non_input_nodes)
    num_non_input_nodes = len(non_input_nodes)

    # Init A
    A: List[List[List[Tuple[Any,Any,Any]]]] = []
    for i in range(num_non_input_nodes):
        row_A = []
        for j in range(num_non_input_nodes):
            row_row_A = []
            for m in range(num_gpus+1):  # dp_gpus
                row_row_A.append((None, None, None))
            row_A.append(row_row_A)
        A.append(row_A)

    # This is calculating T_i_s for last stage.
    network_bandwidth = edge_weight_function.bw
    # Init cumsums
    cum_compute_times = []
    cum_parameter_sizes = []
    cum_sum = 0.0
    cum_activation_size = 0.0
    cum_parameter_size = 0.0
    for i, node in enumerate(work_graph.non_input_nodes):
        cum_sum += node_weight_function(node)
        cum_compute_times.append(cum_sum)
        cum_parameter_size += node.num_parameters
        cum_parameter_sizes.append(cum_parameter_size)

    del cum_sum
    del cum_parameter_size
    del cum_activation_size

    # compute stage time T_i_j
    for i, _ in range(num_non_input_nodes):
        for j in range(i+1, num_non_input_nodes):  # can do it i+1 but whatever
            comp_time = cum_compute_times[j] - cum_compute_times[i]
            param_size = cum_parameter_sizes[j] - cum_parameter_sizes[i]

            for m in range(1,num_machines+1):
                # TODO: replace this with "T_stage"...
                comp_time, data_parallel_communication_time = replication(dp_workers=m, parameter_size=param_size,
                                                                  network_bandwidth=network_bandwidth, compute_time=comp_time,
                                                                  edge_weight_function=edge_weight_function)
                # TODO: what is this? for 1 machine: its
                # TODO: pipedream has errors here.
                # data_parallel_communication_time = (4 * j * cum_parameter_size) / (network_bandwidth * (j + 1))
                # TODO: divide by num machines per machine in case the DDP is smart
                if comp_time == 0:
                    A[i][j][m] = (None, None, None)
                else:
                    A[i][j][m] = (comp_time + data_parallel_communication_time, None, m)


    min_outputid = min(work_graph.output_ids)

    def is_valid_lower_bounds(division):
        return division[-1] < min_outputid

    input_change_indices = list(range(num_non_input_nodes))

    # partitions_n = num_gpus

    def divisions(partitions_n, last_layer):
        # assuming first layer is 0
        for division in combinations(input_change_indices[:last_layer], partitions_n - 1):
            if is_valid_lower_bounds(division):
                yield division

    def get_division_compute_times(division, last_layer=-1):
        # assuming first layer == 0
        if not division:  # data-parallel only
            return [cum_compute_times[last_layer]]

        compute_times = []
        prev = 0.0

        for i in division:
            t = cum_compute_times[i]
            compute_times.append(t - prev)
            prev = t

        compute_times.append(cum_compute_times[last_layer] - prev)  # remaining: to last stage
        return compute_times

    def T_stage(first_layer, last_layer, dp_workers):
        if first_layer is None:
            start = 0.0
        else:
            start = cum_compute_times[first_layer]

        end = cum_compute_times[last_layer]
        total_compute = end - start

        # stage_nodes = non_input_nodes[first_layer:last_layer + 1]
        parameter_size = cum_parameter_sizes[last_layer] - cum_parameter_sizes[first_layer]
        network_bandwidth = cwf.ewf.bw

        compute_time, dp_communication_time = replication(dp_workers=dp_workers, parameter_size=parameter_size,
                                                          network_bandwidth=network_bandwidth,
                                                          compute_time=total_compute,
                                                          edge_weight_function=edge_weight_function)

        # T_i_j = max(compute_time, dp_communication_time)

        T_i_j = compute_time + dp_communication_time
        return T_i_j

    def get_stage_pipeline_comm_time(first_layer, last_layer, dp_workers):

        stage_nodes = non_input_nodes[first_layer:last_layer + 1]

        outgoing_edges, outgoing_nodes, incoming_edges, incoming_nodes = CoarsenedWeightFunction.calculate_borders(
            stage_nodes)

        comm_bwd, comm_fwd = cwf.calculate_comm_forward_and_backward(incomming_edges=incoming_edges,
                                                                     outgoing_edges=outgoing_edges)

        # FIXME: explicitly putting pipedream's modeling errors here.
        param_comm_bwd = 0.0
        for (a, b) in incoming_edges:
            if 'Parameter' in a.scope and a.req_grad:  # TODO: is_parameter, is_param
                param_comm_bwd += cwf.calculate_comm_backward([(a, b)])
        param_comm_fwd = 0.0
        for (a, b) in outgoing_edges:
            if a.va and a.req_grad:
                param_comm_fwd += cwf.calculate_comm_forward([(a, b)])

        # FIXME: (pipedream orig error) this is not correct, since we ***broadcast***. we need to know the number of GPUs in the prev/next stage
        comm_fwd = param_comm_fwd + (comm_fwd - param_comm_fwd) / dp_workers
        comm_bwd = param_comm_bwd + (comm_bwd - param_comm_bwd) / dp_workers

        # FIXME: explicitly putting pipedream's modeling errors here.
        # I have the correct computation in cwf
        pipeline_comm = max(2 * comm_fwd, 2 * comm_bwd)

        return pipeline_comm



    # todo
    # todo
    # todo
    # todo
    # todo
    # todo

    for m in range(1, num_gpus+1):
        for j in range(1, num_non_input_nodes):
            for s in range(j):
                for dp_m in range(1, m+1):
                    pp_m = m-dp_m
                    for division in divisions(partitions_n=pp_m, last_layer=s):
                        compute_times = get_division_compute_times(division, last_layer=s)
                        get_stage_pipeline_comm_time(first_layer=j + 1)

    # def optimial_pipe_j(first_layer, last_layer, pp_workers):
    #     dp_workers = num_gpus - pp_workers
    #     get_stage_pipeline_comm_time(first_layer, last_layer, )
    #
    #     assert first_layer
    i=0
    for j in range(i+1, num_non_input_nodes):
        # num machines loop: stay same.
        for pp_m in range(1, num_gpus):
            dp_m = num_gpus - pp_m
            for division in divisions(partitions_n=pp_m, last_layer=j):
                compute_times = get_division_compute_times(division, last_layer=j)
                get_stage_pipeline_comm_time(first_layer=j+1)



def replication(dp_workers, parameter_size, network_bandwidth, compute_time, edge_weight_function: EdgeWeightFunction):
    if dp_workers == 0:
        raise NotImplementedError()

    bytes_per_param = 4
    dp_communication_time = (bytes_per_param * (dp_workers - 1) * parameter_size) / (network_bandwidth * (dp_workers))
    if compute_time is not None:
        compute_time /= dp_workers
    dp_communication_time /= dp_workers
    dp_communication_time *= edge_weight_function.MULT_FACTOR
    return compute_time, dp_communication_time



def all_predecessors(graph: Graph):
    all_predecessor_ids = dict()

    for node in graph.nodes:
        all_predecessor_ids[node.id] = {n.id for n in node.in_edges}
        for n in node.in_edges:
            all_predecessor_ids[node.id].update(all_predecessor_ids[n.id])

    return all_predecessor_ids