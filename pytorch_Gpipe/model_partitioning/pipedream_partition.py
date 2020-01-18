
import math
from ..model_profiling import Graph, NodeTypes
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
                stashed_data_size /= math.ceil((num_machines - (m+1)) / (m+1))
                if stashed_data_size > memory_size:
                    continue
                # dp sync time
                dp_comm = (4 * m * cum_parameter_size) / (bandwidth * (m+1))
                dp_comm /= num_machines_within_machine

                # A should be upper diagonal they just hold the full matrix if j < i it's invalid (same as i->j)
                if cum_compute_time is None:
                    A[i][j][m] = (None, None, None)
                else:
                    # for some reason they do not take a max here but sum
                    stage_time = (cum_compute_time + dp_comm) / (m+1)
                    A[i][j][m] = (stage_time, None, (m+1))

    min_machines = 1
    max_i = len(compute_times) if not final_level else 1
    # TODO not sure what final level means

    # find the optimal pipeline
    # scan order is depth wise
    for i in range(max_i):
        for m in range(min_machines, num_machines):
            for j in range(i+1, len(compute_times[0])):
                (min_pipeline_time, optimal_split,
                 optimal_num_machines) = A[i][j][m]

                # if we enable using less machines than given and using less machines is better
                # embrace that solution
                if use_fewer_machines and m > 0 and (min_pipeline_time is None or A[i][j][m-1][0] < min_pipeline_time):
                    (min_pipeline_time, optimal_split,
                     optimal_num_machines) = A[i][j][m-1]

                # aggregate results from previous calculations
                # optimal pipelines using layers i->k
                for k in all_predecessor_ids[j]:
                    # looks like a loop k->i->k so we do not aggregate results
                    if i > 0 and k in all_predecessor_ids[i-1]:
                        continue

                    # loop over possible machine configs
                    max_m_prime = 2 if straight_pipeline else (m+1)
                    for m_prime in range(1, max_m_prime):

                        # set input recive time/gradient send time
                        input_transfer_time = 2.0 * output_activation_sizes[k]
                        input_transfer_time /= (bandwidth * m_prime)

                        output_transfer_time = None
                        # calculate output transfer/gradient recive for the last layer
                        if j < len(output_activation_sizes) - 1:
                            output_transfer_time = 2*output_activation_sizes[j]
                            output_transfer_time /= (bandwidth * m_prime)

                        if compute_times[k+1][j] is None:
                            continue

                        # check if within memory constraints
                        last_stage_parameter_size = parameter_sizes[k+1][j]
                        stashed_data_size = activation_sizes[k+1][j]
                        stashed_data_size += last_stage_parameter_size
                        stashed_data_size *= math.ceil(
                            (num_machines - (m+1)) / m_prime)
                        if stashed_data_size > memory_size:
                            continue

                        # compute time of the last stage
                        last_stage_comm *= 4 * (m_prime - 1)
                        last_stage_comm /= (bandwidth * m_prime)
                        last_stage_time = compute_times[k + 1][j]
                        last_stage_time = last_stage_parameter_size
                        last_stage_time += last_stage_comm
                        last_stage_time /= m_prime

                        if A[i][k][m-m_prime][0] is None:
                            continue

                        # find the slowest stage time for this config
                        pipeline_time = max(A[i][k][m-m_prime][0],
                                            last_stage_time)
                        input_transfer_time /= activation_compression_ratio
                        if output_transfer_time is not None:
                            output_transfer_time /= activation_compression_ratio
                            pipeline_time = max(
                                pipeline_time, output_transfer_time)
                        pipeline_time = max(pipeline_time, input_transfer_time)
                        # if we've found a better config emrace it
                        if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                            optimal_split = (k, m-m_prime)
                            optimal_num_machines = m_prime
                            min_pipeline_time = pipeline_time

                # set optimal results
                A[i][j][m] = (min_pipeline_time, optimal_split,
                              optimal_num_machines)

    return A


def cumalative_stats(graph: Graph):
    input_sizes = dict()
    output_sizes = dict()
    stage_size = dict()
    stage_forward_time = dict()
    stage_backward_time = dict()
    for i in range(len(graph)):
        for j in range(i, len(graph)):
            size = stage_size.get((i, j))[0]
            stage_size[(i, j)] = size + _get_size(graph.nodes[i])

            forward_time = stage_forward_time.get((i, j))[0]
            stage_forward_time[(i, j)] = forward_time + \
                _get_time(graph.nodes[i], forward=True)

            backward_time = stage_backward_time.get((i, j))[0]
            stage_backward_time[(i, j)] = backward_time + \
                _get_time(graph.nodes[i], forward=False)

            inputs = input_sizes.get((i, j), set())
            inputs.update(graph.nodes[i].in_nodes)

            output_sizes[(i, j)] = 0
            for out in graph.nodes[i].out_nodes:
                if out.idx > j:
                    output_sizes[(i, j)] += _get_output(graph.nodes[i])

    return input_sizes, output_sizes, stage_size, stage_forward_time, stage_backward_time


def _get_size(node):
    if node.type is NodeTypes.LAYER:
        return node.weight.layer_size
    elif node.type in [NodeTypes.IN, NodeTypes.BUFF_PARAM]:
        return node.weight
    else:
        # TODO maybe base on predecessor
        return 0


def _get_time(node, forward=True):
    if node.type is NodeTypes.LAYER:
        return node.weight.forward_time if forward else node.weight.backward_time
    elif node.type is NodeTypes.OP:
        # TODO maybe base on predecessor
        return 0
    else:
        return 0


def _get_output(node):
    # TODO return output maybe for ops alse
    if node.type is NodeTypes.LAYER:
        return node.weight.output_size
    else:
        return 0
