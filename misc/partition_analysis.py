import torch
from collections import deque
import time
import numpy as np
from .ssgd_analysis import run_analysis as ssgd_run_analysis
from pprint import pprint
import sys
import io
from contextlib import redirect_stdout
import warnings


def run_analysis(sample,
                 graph,
                 config,
                 n_iter,
                 recomputation=True,
                 bw_GBps=12,
                 verbose=True,
                 async_pipeline=False,
                 add_comm_times_to_balance=True,
                 sequential_model=None):

    # NOTE: add_comm_times_to_balance, should be true...
    # setting to false is mainly for our own debug

    TOPO_AWARE = False
    UTILIZATION_SLOWDOWN_SPEEDUP = True
    PRINT_THEORETICAL = False
    PRINT_VAR_STD = False
    TRY_SSGD_ANALYSIS = True

    if graph is not None:
        # thoeretical analysis
        sequential_f, sequential_b, parallel_f, parallel_b = theoretical_analysis(
            graph,
            config,
            recomputation=recomputation,
            async_pipeline=async_pipeline)
        edges = edge_cut(graph)
        # theoretical analysis based on the graph assuming the computation is sequential
        theoretical_sequential_b_balance = worst_balance(sequential_b)
        theoretical_sequential_f_balance = worst_balance(sequential_f)
        (topology_aware_sequential_f_balance,
         topology_aware_sequential_b_balance) = topology_aware_balance(
             sequential_f, sequential_b, edges)
        # theoretical anaysis based on the graph assuming the computation is fully parallel
        theoretical_parallel_b_balance = worst_balance(parallel_b)
        theoretical_parallel_f_balance = worst_balance(parallel_f)
        topology_aware_parallel_f_balance, topology_aware_parallel_b_balance = topology_aware_balance(
            parallel_f, parallel_b, edges)
    else:
        edges = None
        TOPO_AWARE = False
        PRINT_THEORETICAL = False

    # real statistics based on generated partitions
    ((real_f_times, f_vars, f_deviance), (real_b_times, b_vars, b_deviance),
     comm_volume_str, comm_volume_stats, nocomm_real_f_times,
     nocomm_real_b_times, warnings_list) = profile_execution(
         sample,
         config,
         n_iter + 1,
         recomputation=recomputation,
         bw_GBps=bw_GBps,
         async_pipeline=async_pipeline,
         add_comm_times_to_balance=add_comm_times_to_balance)

    real_b_balance = worst_balance(real_b_times)
    real_f_balance = worst_balance(real_f_times)

    if TOPO_AWARE:
        (topology_aware_real_f_balance,
         topology_aware_real_b_balance) = topology_aware_balance(
             real_f_times, real_b_times, edges)

    real_b_slowdown = slowdown(real_b_times, nocomm_real_b_times)
    real_f_slowdown = slowdown(real_f_times, nocomm_real_f_times)

    # NOTE: can also print imbalance slowdown.

    comp_comm_ratio_f = computation_communication_ratio(
        nocomm_real_f_times,
        {k: v['send time']
         for k, v in comm_volume_stats.items()})

    comp_comm_ratio_b = computation_communication_ratio(
        nocomm_real_b_times,
        {k: v['recieve_time']
         for k, v in comm_volume_stats.items()})

    real_f_utilization = utilization(real_f_times, comp_comm_ratio_f)
    real_b_utilization = utilization(real_b_times, comp_comm_ratio_b)

    n_partitions = sum([1 for k in config if isinstance(k, int)])
    expected_speedup = expected_speedup_after_partitioning(
        real_f_times, real_b_times, nocomm_real_f_times, nocomm_real_b_times)

    def rounddict(d, x=2):
        return {k: round(v, x) for k, v in d.items()}

    comp_comm_ratio_f = rounddict(comp_comm_ratio_f)
    comp_comm_ratio_b = rounddict(comp_comm_ratio_b)

    real_b_utilization = rounddict(real_b_utilization)
    real_f_utilization = rounddict(real_f_utilization)

    # TODO: save this into some data structure
    # where we could analyze it later, compare between partitions, etc.
    if verbose:
        s = "-I- Printing Report\n"
        if warnings_list:
            s += "warnings:\n" + "\n".join(warnings_list)

        if edges is not None:
            s += "cutting edges are edges between partitions\n"
            s += f"number of cutting edges: {len(edges)}\n\n"

        s += f"backward times {'do not ' if not recomputation else ''}include recomputation\n"
        if async_pipeline and recomputation:
            s += f"Analysis for async_pipeline=True: last partition will not do recomputation.\n"
        if PRINT_THEORETICAL:
            s += f"\ntheoretical times are execution time based on sum of graph weights ms\n"
            s += f"\nsequential forward {sequential_f}\nsequential backward {sequential_b}\n"
            s += f"parallel forward {parallel_f}\nparallel backward {parallel_b}\n"

        s += f"\nreal times are based on real measurements of execution time of generated partitions ms\n"

        s += f"forward {rounddict(real_f_times)}\nbackward {rounddict(real_b_times)}\n"
        if PRINT_VAR_STD:
            s += f"variance of real execution times ms\n"
            s += f"forward{rounddict(f_vars)}\nbackward{rounddict(b_vars)}\n"

            s += f"avg diviation from the mean of real execution times ms\n"
            s += f"forward{rounddict(f_deviance)}\nbackward{rounddict(b_deviance)}\n"

        s += "\nbalance is ratio of computation time between fastest and slowest parts."
        s += " (between 0 and 1 higher is better)\n"
        if PRINT_THEORETICAL:
            s += f"theoretical sequential balance:\n"
            s += f"forward {theoretical_sequential_f_balance:.3f}\nbackward {theoretical_sequential_b_balance:.3f}\n"
            s += f"theoretical parallel balance:\n"
            s += f"forward {theoretical_parallel_f_balance:.3f}\nbackward {theoretical_parallel_b_balance:.3f}\n"

        s += f"\nreal balance:\n"
        s += f"forward {real_f_balance:.3f}\nbackward {real_b_balance:.3f}\n"

        if TOPO_AWARE:
            s += "\ntopology aware balance is worst balance between 2 connected partitions\n"
            s += f"theoretical sequential topology aware balance:\n"
            s += f"forwad {topology_aware_sequential_f_balance:.3f}\n"
            s += f"backward {topology_aware_sequential_b_balance:.3f}\n"
            s += f"theoretical parallel topology aware balance:\n"
            s += f"forwad {topology_aware_parallel_f_balance:.3f}\n"
            s += f"backward {topology_aware_parallel_b_balance:.3f}\n"

            s += f"\nreal topology aware balance:\n"
            s += f"forwad {topology_aware_real_f_balance:.3f}\nbackward {topology_aware_real_b_balance:.3f}\n"

        s += f"\nAssuming bandwidth of {bw_GBps} GBps between GPUs\n"
        s += f"\ncommunication volumes size of activations of each partition\n"
        for idx, volume in comm_volume_str.items():
            s += f"{idx}: {volume}\n"

        s += "\nCompuatation Communication ratio (comp/(comp+comm)):\n"
        s += f"forward {comp_comm_ratio_f} \nbackward {comp_comm_ratio_b}\n"

        if UTILIZATION_SLOWDOWN_SPEEDUP:
            s += "\nPipeline Slowdown: (compared to sequential executation with no communication)\n"
            s += f"forward {real_f_slowdown:.3f}\nbackward {real_b_slowdown:.3f}\n"

            s += "\nExpected utilization by partition\n"
            s += f"forward {real_f_utilization}\nbackward {real_b_utilization}\n"

            s += f"\nExpected speedup for {n_partitions} partitions is: {expected_speedup:.3f}"

        if TRY_SSGD_ANALYSIS and torch.cuda.is_available() and (
                sequential_model is not None):
            n_workers = n_partitions
            model = sequential_model
            try:
                ssgd_expected_speedup, ssgd_stats = ssgd_run_analysis(
                    sample, model, n_workers, bw_GBps=bw_GBps, verbose=verbose)
                # except Exception as e:
                if verbose:
                    ssgd_output = None
                    with io.StringIO() as buf, redirect_stdout(buf):
                        print()
                        print('Printing SSGD analysis:')
                        print(
                            "(naive: assuming 0 concurency between communication and computation)"
                        )
                        pprint(rounddict(ssgd_stats))
                        print(
                            f"ssgd_expected_speedup: {ssgd_expected_speedup:.3f}"
                        )
                        pipeline_to_ssgd_speedup = expected_speedup / ssgd_expected_speedup
                        print(f"Pipeline/SSGD: {pipeline_to_ssgd_speedup:.3f}")
                        ssgd_output = buf.getvalue()

                    print(ssgd_output)

            except:
                print(f"SSGD analysis failed: {sys.exc_info()[0]}")
                # raise

        print(s)

    return expected_speedup, s  # real_f_balance, real_b_balance


#################################
# analyze generated partitions
# ##############################


def profile_execution(model_inputs,
                      partition_config,
                      n_iters,
                      recomputation=True,
                      bw_GBps=12,
                      async_pipeline=False,
                      add_comm_times_to_balance=True):
    '''perfrom forward/backward passes and measure execution times accross n batches
    '''
    n_partitions = sum([1 for k in partition_config if isinstance(k, int)])
    f_times = {i: [] for i in range(n_partitions)}
    b_times = {i: [] for i in range(n_partitions)}

    nocommf_times = {i: [] for i in range(n_partitions)}
    nocommb_times = {i: [] for i in range(n_partitions)}

    communication_volume = {}
    communication_stats = {}
    is_parameter = set()
    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    # Return warnings so we can print
    warnings_list = []

    for current_iteration_num in range(n_iters):
        parts = deque(range(n_partitions))
        activations = {}
        for i, t in zip(partition_config['model inputs'], model_inputs):
            # save activations on CPU in order to save GPU memory
            activations[i] = t.cpu()

        # TODO: make it just a forward pass, then do a backward pass. (Will allow handling nested tuples)
        # perform one run of the partitions
        while len(parts) > 0:
            idx = parts.popleft()

            # For async pipeline, do no use recomputation on last partition
            is_last_partition = (len(parts) == 0)
            is_first_partition = (idx == 0)
            partition_specific_recomputation = recomputation
            if async_pipeline and is_last_partition:
                partition_specific_recomputation = False

            # partition_specific_inputs_requires_grad
            inputs_requires_grad = not is_first_partition

            if all(tensor in activations
                   for tensor in partition_config[idx]['inputs']):
                inputs = []
                for tensor in partition_config[idx]['inputs']:
                    t = activations[tensor]
                    # shared weights support
                    if tensor in is_parameter:
                        t.requires_grad_()
                    inputs.append(t)

                # input statistics
                in_size_mb = sum([(t.nelement() * t.element_size())
                                  for t in inputs]) / 1e6
                recv_time = in_size_mb / bw_GBps
                # time measurement
                if torch.cuda.is_available():
                    f_time, b_time, outputs = cuda_time(
                        partition_config[idx]['model'],
                        inputs,
                        recomputation=partition_specific_recomputation,
                        inputs_requires_grad=inputs_requires_grad)
                else:
                    f_time, b_time, outputs = cpu_time(
                        partition_config[idx]['model'],
                        inputs,
                        recomputation=partition_specific_recomputation,
                        inputs_requires_grad=inputs_requires_grad)

                # output statistics
                out_size_mb = 0
                send_time = 0
                for o, t in zip(partition_config[idx]['outputs'], outputs):
                    # Check if contations
                    if current_iteration_num == 0:
                        if not t.is_contiguous():
                            warnining = f"Partition{idx} output:{o} is not contiguous!"
                            warnings.warn(warnining)
                            warnings_list.append(warnining)

                    # save activation on CPU in order to save GPU memory
                    if isinstance(t, torch.nn.Parameter):
                        # shared weights support
                        is_parameter.add(o)

                    activations[o] = t.detach().cpu()
                    t_mb = (t.nelement() * t.element_size()) / 1e6
                    t_send = t_mb / bw_GBps
                    out_size_mb += t_mb
                    send_time += t_send

                del outputs

                if is_last_partition:
                    send_time = 0.0

                stats = {
                    "input size": in_size_mb,  # "MB "
                    "recieve_time": recv_time,  # "ms"
                    "out": out_size_mb,  # "MB"
                    "send time": send_time,  # ms"
                }

                units = {
                    "input size": "MB",
                    "recieve_time": "ms",
                    "out": "MB",
                    "send time": "ms",
                }
                newd = {k: f"{stats[k]:.2f} {units[k]}" for k in stats}
                communication_volume[idx] = ', '.join(
                    "{!s}:{!r}".format(key, val)
                    for (key, val) in newd.items())

                communication_stats[idx] = stats

                # Adding communication time to balance:
                # time = time + comm_send

                nocommf_times[idx].append(f_time)
                nocommb_times[idx].append(b_time)

                if add_comm_times_to_balance:
                    if not is_last_partition:
                        f_time += send_time
                    if not is_first_partition:
                        b_time += in_size_mb / bw_GBps  # HACK: activation input = gradient size

                f_times[idx].append(f_time)
                b_times[idx].append(b_time)

            else:
                parts.append(idx)

    # calculate mean and variance
    return mean_var(f_times), mean_var(
        b_times), communication_volume, communication_stats, mean_var(
            nocommf_times)[0], mean_var(nocommb_times)[0], warnings_list


def mean_var(times):
    means = dict()
    variances = dict()
    avg_deviations = dict()
    for i, ts in times.items():
        max_v = max(ts)
        arr = np.array([t for t in ts if t < max_v])
        means[i] = np.mean(arr)
        variances[i] = np.var(arr)
        avg_deviations[i] = np.abs((arr - means[i])).mean()

    return means, variances, avg_deviations


def cuda_time(partition,
              inputs,
              recomputation=True,
              inputs_requires_grad=False):
    # now we move partition to GPU
    partition = partition.to('cuda')
    partition.device = 'cuda'
    b_time = cuda_backward(partition,
                           inputs,
                           recomputation=recomputation,
                           inputs_requires_grad=inputs_requires_grad)

    # Delete gradeinets to save space
    for p in partition.parameters():
        p.grad = None

    f_time, outputs = cuda_forward(partition,
                                   inputs,
                                   recomputation=recomputation)
    partition = partition.cpu()
    partition.device = 'cpu'
    return f_time, b_time, outputs


def cuda_backward(partition,
                  inputs,
                  recomputation=True,
                  inputs_requires_grad=False):
    ''' measure forward/backward time of a partition on the GPU
    '''
    # now we move inputs to GPU
    inputs = [
        i.to('cuda').requires_grad_(inputs_requires_grad
                                    and i.is_floating_point()) for i in inputs
    ]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if recomputation:
        torch.cuda.synchronize(device='cuda')
        start.record()
        outputs = partition(*inputs)
    else:
        outputs = partition(*inputs)
        torch.cuda.synchronize(device='cuda')
        start.record()
    loss = sum(o.norm() for o in outputs)  # FIXME: just use real loss.
    loss.backward()
    end.record()
    torch.cuda.synchronize(device='cuda')
    b_time = (start.elapsed_time(end))

    return b_time


def cuda_forward(partition, inputs, recomputation=True):
    # now we move inputs to GPU
    inputs = [i.to('cuda') for i in inputs]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device='cuda')
    with torch.set_grad_enabled(not recomputation):
        start.record()
        outputs = partition(*inputs)
        end.record()
        torch.cuda.synchronize(device='cuda')
        f_time = (start.elapsed_time(end))
    return f_time, outputs


def cpu_time(partition,
             inputs,
             recomputation=True,
             inputs_requires_grad=False):
    ''' measure forward/backward time of a partition on the CPU
    '''
    partition = partition.to('cpu')
    partition.device = 'cpu'
    b_time = cpu_backward(partition,
                          inputs,
                          recomputation=recomputation,
                          inputs_requires_grad=inputs_requires_grad)
    f_time, outputs = cpu_forward(partition,
                                  inputs,
                                  recomputation=recomputation)

    return f_time, b_time, outputs


def cpu_forward(partition, inputs, recomputation=True):
    inputs = [i.cpu() for i in inputs]
    with torch.set_grad_enabled(not recomputation):
        start = time.time()
        outputs = partition(*inputs)
        end = time.time()
        f_time = 1000 * (end - start)

    return f_time, outputs


def cpu_backward(partition,
                 inputs,
                 recomputation=True,
                 inputs_requires_grad=False):
    inputs = [
        i.cpu().requires_grad_(inputs_requires_grad and i.is_floating_point())
        for i in inputs
    ]
    start = time.time()
    outputs = partition(*inputs)
    if not recomputation:
        start = time.time()
    loss = sum(o.norm() for o in outputs)
    loss.backward()
    end = time.time()
    b_time = 1000 * (end - start)

    return b_time


###################################
# analysis based on the graph
# ##################################
def edge_cut(graph):
    '''
    find the cutting edges of the graph
    '''
    edges = []
    for n in graph.nodes:
        for u in n.out_nodes:
            if n.part != u.part:
                edges.append((n, u))

    return edges


def theoretical_analysis(graph,
                         partition_config,
                         recomputation=True,
                         async_pipeline=False):
    ''' find execution time of partitions based on the model's graph using 2 a sequential assumption and parallel assumption
        the sequential assumption is that in the partition all operation are linear.
        the parallel assumption assumes that all computation paths are concurrent.
    '''
    n_parts = len(set(n.part for n in graph.nodes))
    parallel_b = dict()
    parallel_f = dict()

    tensor_names = set()
    for i in range(n_parts):
        tensor_names.update(partition_config[i]['outputs'])

    sequential_f = {i: 0 for i in range(n_parts)}
    sequential_b = {i: 0 for i in range(n_parts)}

    nodes = dict()
    for node in graph.nodes:
        # cache relevant nodes to make fetching them faster
        if node.scope in tensor_names:
            nodes[node.scope] = node

        # old way of measuring time as sum of all computation
        sequential_f[node.part] += extract_time(node.weight, forward=True)
        sequential_b[node.part] += extract_time(node.weight, forward=False)

    # new way of measuring time as longest path where all paths are concurrent
    for i in range(n_parts):
        partition_sepsific_recomputation = recomputation
        is_last_partition = (i == n_parts - 1)
        if async_pipeline and is_last_partition:
            partition_sepsific_recomputation = False

        outputs = [nodes[name] for name in partition_config[i]['outputs']]
        cache = dict()
        parallel_f[i] = 0
        parallel_b[i] = 0
        for o in outputs:
            f, b = parallel_execution_analysis(o, i, cache)
            parallel_f[i] = max(parallel_f[i], f)
            parallel_b[i] = max(parallel_b[i], b)

        if partition_sepsific_recomputation:
            sequential_b[i] += sequential_f[i]
            parallel_b[i] += parallel_f[i]

    return sequential_f, sequential_b, parallel_f, parallel_b


def parallel_execution_analysis(node, part_idx, cache):
    # use cache in order to remember common subpaths
    if node.scope in cache:
        return cache[node.scope]
    elif node.part != part_idx:
        cache[node.scope] = (0, 0)
        return 0, 0

    longest_f, longest_b = 0, 0

    for n in node.in_nodes:
        f, b = parallel_execution_analysis(n, part_idx, cache)
        longest_f = max(f, longest_f)
        longest_b = max(b, longest_b)

    longest_f += extract_time(node.weight, forward=True)
    longest_b += extract_time(node.weight, forward=False)

    cache[node.scope] = (longest_f, longest_b)

    return longest_f, longest_b


def extract_time(w, forward=False):
    if hasattr(w, "weight"):
        w = w.weight
    if not hasattr(w, "forward_time"):
        return 0
    if forward:
        return w.forward_time
    return w.backward_time


####################################
# balance computation
# ##################################


def computation_communication_ratio(comp_times, comm_times):

    # comm_times = {k: v['send time'] for k, v in comm_times.items()}
    # comm_times = {k: v['recieve_times'] for k, v in comm_times.items()}

    assert (len(comp_times) == len(comm_times))
    ratio = {
        k: comp_times[k] / (comm_times[k] + comp_times[k])
        for k in comp_times
    }
    return ratio


def utilization(times, comp_fraction):
    # TODO: I still think this statistic can be improved... its just an estimation.

    worst = max(times.values())
    # This assumes that the GPU is utilized while we do comunication. (but its generally not)
    base_util = {k: round(v / worst, 2) for k, v in times.items()}

    # Therefore we mutiply by comp fraction
    comp_util = {k: base_util[k] * comp_fraction[k] for k in comp_fraction}
    return comp_util


def slowdown(times, times_wo_comm):

    worst = max(times.values())
    n_partitions = len(times)

    ideal = sum(times_wo_comm.values())
    actual = n_partitions * worst

    model_parallel_and_partitioning_slowdown = actual / ideal

    return model_parallel_and_partitioning_slowdown


def imbbalance_slowdown(times):
    worst = max(times.values())
    n_partitions = len(times)

    total = sum(times.values())
    actual = n_partitions * worst

    partitioning_slowdown = actual / total

    # NOTE: Expected speedup for X accelerators:
    #  Expected_speedup = sum(times.values()) / worst
    # # So, we should optimize towards lowering the worstcase as much as possible.
    # expected_speedup = n_partitions / partitioning_slowdown

    return partitioning_slowdown


def expected_speedup_after_partitioning(fwd_times, bwd_times,
                                        fwd_times_wo_comm, bwd_times_wo_comm):

    n_partitions = len(fwd_times)
    assert (len(fwd_times) == len(bwd_times))

    fwd_slowdown = slowdown(fwd_times, fwd_times_wo_comm)
    bwd_slowdown = slowdown(bwd_times, bwd_times_wo_comm)

    worst_fwd = max(fwd_times.values())
    worst_bwd = max(bwd_times.values())
    fwd_plus_bwd = worst_fwd + worst_bwd

    bwd_ratio = worst_bwd / fwd_plus_bwd
    fwd_ratio = worst_fwd / fwd_plus_bwd

    partitioning_slowdown = (bwd_ratio * bwd_slowdown) + (fwd_ratio *
                                                          fwd_slowdown)

    #  Expected speedup for X accelerators:
    #  NOTE: Expected_speedup = sum(times.values()) / worst
    # So, we should optimize towards lowering the worstcase as much as possible.
    expected_speedup = n_partitions / partitioning_slowdown

    return expected_speedup


def worst_balance(times):
    return min(times.values()) / max(times.values())


def topology_aware_balance(f_times, b_times, cutting_edges):
    ''' find the lowest balance between 2 connected partitions
    '''
    f_balance = b_balance = 10
    for u, v in cutting_edges:
        f_ratio = min(f_times[u.part], f_times[v.part]) / \
            max(f_times[u.part], f_times[v.part])

        b_ratio = min(b_times[u.part], b_times[v.part]) / \
            max(b_times[u.part], b_times[v.part])

        if f_ratio < f_balance:
            f_balance = f_ratio

        if b_ratio < b_balance:
            b_balance = b_ratio

    return f_balance, b_balance


######################
# unused code
# ####################
def run_partitions(model_inputs, partition_config):
    n_partitions = sum([1 for k in partition_config if isinstance(k, int)])

    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    activations = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(n_partitions):
        partition_config[i]['model'] = partition_config[i]['model'].to(device)
        partition_config[i]['model'].device = device

    for i, t in zip(partition_config['model inputs'], model_inputs):
        activations[i] = t.to(device)

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        # if all inputs are ready run partition
        if all(tensor in activations
               for tensor in partition_config[idx]['inputs']):
            inputs = [
                activations[tensor]
                for tensor in partition_config[idx]['inputs']
            ]
            outs = partition_config[idx]['model'](*inputs)
            for o, t in zip(partition_config[idx]['outputs'], outs):
                activations[o] = t
        else:
            parts.append(idx)

    return [activations[o] for o in partition_config['model outputs']]
