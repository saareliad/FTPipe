from typing import Optional
import math
from ..model_profiling import Graph, NodeTypes, NodeWeightFunction, EdgeWeightFunction, Node

# from the graph creation we have topological sort


# compute_times activations_sizes parameter_sizes are NxN accumalative matrices
# all_predecessor_ids[i] all nodes that are used to compute an input of i
# num_machines total number of servers?
# num_machines_within_machines number of gpus per server?
# bandwidth is comm bandwidth of gpus fixed for all gpus
# straight_pipeline disable replication
# use_fewer_machines enables using less gpus
# memory the size of available gpu memory fixed for all gpus
# activation_compression_ratio the compression ratio if we compress tensors prior to sending them
# final level not sure


# TODO if we take ops into account than a lot of info will be missing (cannot know sizes/times)
# maybe we should do this only on the layers graph aka only inputs and layers and fill the gaps afterwards
# we should note that the layers graph is not optimal and also we will have to to interpolation in order to place ops


def compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                         output_activation_sizes, all_predecessor_ids,
                         num_machines, num_machines_within_machine,
                         straight_pipeline=True, use_fewer_machines=False,
                         bandwidth=None, memory_size=None, activation_compression_ratio=None,
                         final_level=False):

    if bandwidth is None:
        bandwidth = math.inf

    if memory_size is None:
        memory_size = math.inf

    if activation_compression_ratio is None:
        activation_compression_ratio = 1

    # init phase build a matrix of size N x N x num_machines x 3
    # A[i][k][m] = (min_pipeline_time, optimal_split,optimal_num_machines) for a stage that spans i->j replicated on m machines
    A = []
    for i in range(len(compute_times)):
        row_A = []
        for j in range(len(compute_times[0])):
            row_row_A = []
            for m in range(num_machines):
                row_row_A.append((None, None, None))
            row_A.append(row_row_A)
        A.append(row_A)

    # init phase fill the matrix A[i][j][m] = T(i,j,m) compute time for a stage with layers i->j replicated on m machines
    for i in range(len(compute_times)):
        for j in range(i, len(compute_times[0])):
            # cum_whatever[i][j] is sum of whatever spanning layers i->j
            cum_compute_time = compute_times[i][j]
            cum_activation_size = activation_sizes[i][j]
            cum_parameter_size = parameter_sizes[i][j]
            max_m = 1 if straight_pipeline else num_machines
            # check for all replication options
            for m in range(max_m):
                # check if are within memory constraints
                stashed_data_size = cum_activation_size + cum_parameter_size
                stashed_data_size /= math.ceil((num_machines -
                                                (m + 1)) / (m + 1))
                if stashed_data_size > memory_size:
                    continue
                # dp sync time
                dp_comm = (4 * m * cum_parameter_size) / (bandwidth * (m + 1))
                dp_comm /= num_machines_within_machine

                # A should be upper diagonal they just hold the full matrix if j < i it's invalid (same as i->j)
                if cum_compute_time is None:
                    A[i][j][m] = (None, None, None)
                else:
                    # for some reason they do not take a max here but sum
                    stage_time = (cum_compute_time + dp_comm) / (m + 1)
                    A[i][j][m] = (stage_time, None, (m + 1))

    min_machines = 1
    max_i = len(compute_times) if not final_level else 1
    # TODO not sure what final level means

    # find the optimal pipeline
    # scan order is depth wise
    for i in range(max_i):
        for m in range(min_machines, num_machines):
            for j in range(i + 1, len(compute_times[0])):
                (min_pipeline_time, optimal_split,
                 optimal_num_machines) = A[i][j][m]

                # if we enable using less machines than given and using less machines is better
                # embrace that solution
                if use_fewer_machines and m > 0 and (min_pipeline_time is None or A[i][j][m - 1][0] < min_pipeline_time):
                    (min_pipeline_time, optimal_split,
                     optimal_num_machines) = A[i][j][m - 1]

                # aggregate results from previous calculations
                # optimal pipelines using layers i->k
                for k in all_predecessor_ids[j]:
                    # looks like a loop k->i->k so we do not aggregate results
                    if i > 0 and k in all_predecessor_ids[i - 1]:
                        continue

                    # loop over possible machine configs
                    max_m_prime = 2 if straight_pipeline else (m + 1)
                    for m_prime in range(1, max_m_prime):

                        # set input recive time/gradient send time
                        input_transfer_time = 2.0 * output_activation_sizes[k]
                        input_transfer_time /= (bandwidth * m_prime)

                        output_transfer_time = None
                        # calculate output transfer/gradient recive for the last layer
                        if j < len(output_activation_sizes) - 1:
                            output_transfer_time = 2 * \
                                output_activation_sizes[j]
                            output_transfer_time /= (bandwidth * m_prime)

                        if compute_times[k + 1][j] is None:
                            continue

                        # check if within memory constraints
                        last_stage_parameter_size = parameter_sizes[k + 1][j]
                        stashed_data_size = activation_sizes[k + 1][j]
                        stashed_data_size += last_stage_parameter_size
                        stashed_data_size *= math.ceil(
                            (num_machines - (m + 1)) / m_prime)
                        if stashed_data_size > memory_size:
                            continue

                        # compute time of the last stage
                        last_stage_comm *= 4 * (m_prime - 1)
                        last_stage_comm /= (bandwidth * m_prime)
                        last_stage_time = compute_times[k + 1][j]
                        last_stage_time = last_stage_parameter_size
                        last_stage_time += last_stage_comm
                        last_stage_time /= m_prime

                        if A[i][k][m - m_prime][0] is None:
                            continue

                        # find the slowest stage time for this config
                        pipeline_time = max(A[i][k][m - m_prime][0],
                                            last_stage_time)
                        input_transfer_time /= activation_compression_ratio
                        if output_transfer_time is not None:
                            output_transfer_time /= activation_compression_ratio
                            pipeline_time = max(
                                pipeline_time, output_transfer_time)
                        pipeline_time = max(pipeline_time, input_transfer_time)
                        # if we've found a better config emrace it
                        if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                            optimal_split = (k, m - m_prime)
                            optimal_num_machines = m_prime
                            min_pipeline_time = pipeline_time

                # set optimal results
                A[i][j][m] = (min_pipeline_time, optimal_split,
                              optimal_num_machines)

    return A


def prepare_partition_data(graph: Graph, node_weight_function: Optional[NodeWeightFunction], edge_weight_function: Optional[EdgeWeightFunction]):
    all_predecessor_ids = all_predecessors(graph)

    forward_time, backward_time, activation_size, memory, output_size = aggregate_stats(graph, all_predecessor_ids,
                                                                                        node_weight_function, edge_weight_function)


def all_predecessors(graph: Graph):
    all_predecessor_ids = dict()

    for node in graph.nodes:
        all_predecessor_ids[node.id] = {n.id for n in node.in_edges}
        for n in node.in_edges:
            all_predecessor_ids[node.id].update(all_predecessor_ids[n.id])

    return all_predecessor_ids


def aggregate_stats(graph: Graph, all_predecessor_ids, node_weight_function: Optional[NodeWeightFunction], edge_weight_function: Optional[EdgeWeightFunction]):
    states = dict()
    # n^2 algorithm might revisit
    # statistics for the dependencies graph of each node
    for u in graph.nodes:
        states[u.id] = dict()
        states[u.id]["forward_time"] = _get_time(u, forward=True)
        states[u.id]["backward_time"] = _get_time(u, forward=False)
        states[u.id]["activation_size"] = _get_activation_size(u)
        states[u.id]["memory"] = _get_memory(u)
        states[u.id]["output_size"] = _get_output_size(u)

        for v in map(lambda idx: graph.nodes[idx] for idx in all_predecessor_ids[u.id]):
            states[u.id]["forward_time"] += _get_time(v, forward=True)
            states[u.id]["backward_time"] += _get_time(v, forward=False)
            states[u.id]["activation_size"] += _get_activation_size(v)
            states[u.id]["memory"] += _get_memory(v)
            states[u.id]["output_size"] += _get_output_size(v)

    forward_time = dict()
    backward_time = dict()
    activation_size = dict()
    memory = dict()
    output_size = dict()

    # calculate all partial stats
    # n^2 algorithm might revisit
    for u in enumerate(graph.nodes):
        for v in enumerate(graph.nodes):
            if u == 0:
                forward_time[(u, v)] = states[v]["forward_time"]
                backward_time[(u, v)] = states[v]["backward_time"]
                activation_size[(u, v)] = states[v]["activation_size"]
                memory[(u, v)] = states[v]["memory"]
                output_size[(u, v)] = states[v]["output_size"]
            elif v >= u:
                f = states[v]["forward_time"] - states[u - 1]["forward_time"]
                b = states[v]["backward_time"] - states[u - 1]["backward_time"]

                a = states[v]["activation_size"]
                a -= states[u - 1]["activation_size"]

                m = states[v]["memory"] - states[u - 1]["memory"]
                o = states[v]["output_size"] - states[u - 1]["output_size"]

                forward_time[(u, v)] = f
                backward_time[(u, v)] = b
                activation_size[(u, v)] = a
                memory[(u, v)] = m
                output_size[(u, v)] = o

    return forward_time, backward_time, activation_size, memory, output_size


def _get_time(node: Node, forward=False):
    if node.type is NodeTypes.LAYER:
        return node.weight.forward_time if forward else node.weight.backward_time
    if node.type is NodeTypes.OP:
        return 1

    return 0


def _get_activation_size(node: Node):
    return 0


def _get_memory(node: Node):
    if node.type is NodeTypes.LAYER:
        return node.weight.layer_size
    if node.type is NodeTypes.IN or NodeTypes.BUFF_PARAM:
        return node.weight
    return 0


def _get_output_size(node: Node):
    return 0
