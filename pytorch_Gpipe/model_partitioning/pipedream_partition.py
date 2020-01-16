
import math

# from the graph creation we have topological sort


def compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                         output_activation_sizes, all_predecessor_ids,
                         num_machines, num_machines_within_machine,
                         bandwidth, straight_pipeline, use_fewer_machines, use_memory_constraint,
                         memory_size, activation_compression_ratio, final_level=True):

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
                # check wether we can stashj weights
                stashed_data_size = math.ceil((num_machines - (m+1)) / (m+1)) * \
                    (cum_activation_size + cum_parameter_size)
                if use_memory_constraint and stashed_data_size > memory_size:
                    continue
                # dp sync time
                dp_comm = (4 * m * cum_parameter_size) / (bandwidth * (m+1))
                dp_comm /= num_machines_within_machine

                # A should be diagonal they just hold the full matrix if j < i it's invalid (same as i->j)
                if cum_compute_time is None:
                    A[i][j][m] = (None, None, None)
                else:
                    # for some reason they do not take a max here
                    A[i][j][m] = (
                        sum([cum_compute_time, dp_comm]) / (m+1), None, (m+1))

    min_machines = 1
    max_i = len(compute_times) if not final_level else 1
    # not sure what final level means

    # find the optimal pipeline
    for i in range(max_i):
        for m in range(min_machines, num_machines):
            for j in range(i+1, len(compute_times[0])):
                (min_pipeline_time, optimal_split,
                 optimal_num_machines) = A[i][j][m]
                if use_fewer_machines and m > 0 and (
                        min_pipeline_time is None or A[i][j][m-1][0] < min_pipeline_time):
                    (min_pipeline_time, optimal_split,
                     optimal_num_machines) = A[i][j][m-1]
                for k in all_predecessor_ids[j]:
                    if i > 0 and k in all_predecessor_ids[i-1]:
                        continue
                    max_m_prime = 2 if straight_pipeline else (m+1)
                    for m_prime in range(1, max_m_prime):
                        input_transfer_time = (2.0 * output_activation_sizes[k]) / \
                            (bandwidth * m_prime)
                        output_transfer_time = None
                        if j < len(output_activation_sizes) - 1:
                            output_transfer_time = (2.0 *
                                                    output_activation_sizes[j]) / (bandwidth * m_prime)

                        last_stage_time = compute_times[k+1][j]
                        if last_stage_time is None:
                            continue
                        last_stage_parameter_size = parameter_sizes[k+1][j]
                        stashed_data_size = (
                            activation_sizes[k+1][j]) + last_stage_parameter_size
                        stashed_data_size *= math.ceil(
                            (num_machines - (m+1)) / m_prime)
                        if use_memory_constraint and stashed_data_size > memory_size:
                            continue
                        last_stage_time = sum([last_stage_time,
                                               ((4 * (m_prime - 1) *
                                                 last_stage_parameter_size) / (bandwidth * m_prime))])
                        last_stage_time /= m_prime

                        if A[i][k][m-m_prime][0] is None:
                            continue
                        pipeline_time = max(
                            A[i][k][m-m_prime][0], last_stage_time)
                        if activation_compression_ratio is not None:
                            input_transfer_time /= activation_compression_ratio
                            if output_transfer_time is not None:
                                output_transfer_time /= activation_compression_ratio
                            pipeline_time = max(
                                pipeline_time, input_transfer_time)
                            if output_transfer_time is not None:
                                pipeline_time = max(
                                    pipeline_time, output_transfer_time)
                        if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                            optimal_split = (k, m-m_prime)
                            optimal_num_machines = m_prime
                            min_pipeline_time = pipeline_time
                A[i][j][m] = (min_pipeline_time, optimal_split,
                              optimal_num_machines)

    return A
