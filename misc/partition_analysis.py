import torch
from collections import deque
import time
import numpy as np


def run_analysis(sample,
                 graph,
                 config,
                 n_iter,
                 recomputation=True,
                 bandwidth_gps=12,
                 verbose=True):
    # thoeretical analysis
    sequential_f, sequential_b, parallel_f, parallel_b = theoretical_analysis(
        graph, config, recomputation=recomputation)
    edges = edge_cut(graph)
    # theoretical analysis based on the graph assuming the computation is sequential
    theoretical_sequential_b_imbalance = worst_imbalance(sequential_b)
    theoretical_sequential_f_imbalance = worst_imbalance(sequential_f)
    (topology_aware_sequential_f_imbalance,
     topology_aware_sequential_b_imbalance) = topology_aware_imbalance(
         sequential_f, sequential_b, edges)
    # theoretical anaysis based on the graph assuming the computation is fully parallel
    theoretical_parallel_b_imbalance = worst_imbalance(parallel_b)
    theoretical_parallel_f_imbalance = worst_imbalance(parallel_f)
    topology_aware_parallel_f_imbalance, topology_aware_parallel_b_imbalance = topology_aware_imbalance(
        parallel_f, parallel_b, edges)
    # real statistics based on generated partitions
    ((real_f_times, f_vars, f_deviance), (real_b_times, b_vars, b_deviance),
     comm_volume) = profile_execution(sample,
                                      config,
                                      n_iter + 1,
                                      recomputation=recomputation,
                                      bandwidth_gps=bandwidth_gps)
    real_b_imbalance = worst_imbalance(real_b_times)
    real_f_imbalance = worst_imbalance(real_f_times)
    (topology_aware_real_f_imbalance,
     topology_aware_real_b_imbalance) = topology_aware_imbalance(
         real_f_times, real_b_times, edges)

    # TODO: save this into some data structure
    # where we could analyze it later, compare between partitions, etc.
    if verbose:
        s = "-I- Printing Report\n"
        s += "cutting edges are edges between partitions\n"
        s += f"number of cutting edges: {len(edges)}\n\n"

        s += f"backward times {'do not ' if not recomputation else ''}include recomputation\n"

        s += f"\ntheoretical times are execution time based on sum of graph weights ms\n"
        s += f"\nsequential forward {sequential_f}\nsequential backward {sequential_b}\n"
        s += f"parallel forward {parallel_f}\nparallel backward {parallel_b}\n"

        s += f"\nreal times are based on real measurements of execution time of generated partitions ms\n"
        s += f"forward {real_f_times}\nbackward {real_b_times}\n"
        s += f"variance of real execution times ms\nforward{f_vars}\nbackward{b_vars}\n"

        s += f"avg diviation from the mean of real execution times ms\nforward{f_deviance}\nbackward{b_deviance}\n"

        s += "\nimbalance is ratio of computation time between fastest and slowest parts."
        s += " (between 0 and 1 higher is better)\n"
        s += f"theoretical sequential imbalance:\n"
        s += f"forward {theoretical_sequential_f_imbalance:.3f}\nbackward {theoretical_sequential_b_imbalance:.3f}\n"
        s += f"theoretical parallel imbalance:\n"
        s += f"forward {theoretical_parallel_f_imbalance:.3f}\nbackward {theoretical_parallel_b_imbalance:.3f}\n"

        s += f"\nreal imbalance:\n"
        s += f"forward {real_f_imbalance:.3f}\nbackward {real_b_imbalance:.3f}\n"

        s += "\ntopology aware imbalance is worst imbalance between 2 connected partitions\n"
        s += f"theoretical sequential topology aware imbalance:\n"
        s += f"forwad {topology_aware_sequential_f_imbalance:.3f}\n"
        s += f"backward {topology_aware_sequential_b_imbalance:.3f}\n"
        s += f"theoretical parallel topology aware imbalance:\n"
        s += f"forwad {topology_aware_parallel_f_imbalance:.3f}\n"
        s += f"backward {topology_aware_parallel_b_imbalance:.3f}\n"

        s += f"\nreal topology aware imbalance:\n"
        s += f"forwad {topology_aware_real_f_imbalance:.3f}\nbackward {topology_aware_real_b_imbalance:.3f}\n"

        s += f"\ncommunication volumes size of activations of each partition\n"
        for idx, volume in comm_volume.items():
            s += f"{idx}: {volume}\n"

        print(s)
    


#################################
# analyze generated partitions
# ##############################

def profile_execution(model_inputs,
                      partition_config,
                      n,
                      recomputation=True,
                      bandwidth_gps=12):
    '''perfrom forward/backward passes and measure execution times accross n batches
    '''
    n_partitions = sum([1 for k in partition_config if isinstance(k, int)])
    f_times = {i: [] for i in range(n_partitions)}
    b_times = {i: [] for i in range(n_partitions)}

    communication_volume = {}
    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    for _ in range(n):
        parts = deque(range(n_partitions))
        activations = {}
        for i, t in zip(partition_config['model inputs'], model_inputs):
            # save activations on CPU in order to save GPU memory
            activations[i] = t.cpu()

        # perform one run of the partitions
        while len(parts) > 0:
            idx = parts.popleft()
            if all(tensor in activations
                   for tensor in partition_config[idx]['inputs']):
                inputs = [
                    activations[tensor]
                    for tensor in partition_config[idx]['inputs']
                ]

                # input statistics
                in_size_mb = 0
                recv_time = 0
                for t in inputs:
                    t_mb = (t.nelement() * t.element_size()) / 1e6
                    t_recv = t_mb / bandwidth_gps
                    in_size_mb += t_mb
                    recv_time = max(recv_time, t_recv)

                # time measurement
                if torch.cuda.is_available():
                    f_time, b_time, outputs = cuda_time(
                        partition_config[idx]['model'],
                        inputs,
                        recomputation=recomputation)
                else:
                    f_time, b_time, outputs = cpu_time(
                        partition_config[idx]['model'],
                        inputs,
                        recomputation=recomputation)

                # output statistics
                out_size_mb = 0
                send_time = 0
                for o, t in zip(partition_config[idx]['outputs'], outputs):
                    # save activation on CPU in order to save GPU memory
                    activations[o] = t.detach().cpu()
                    t_mb = (t.nelement() * t.element_size()) / 1e6
                    t_send = t_mb / bandwidth_gps
                    out_size_mb += t_mb
                    send_time = max(t_send, send_time)

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
                newd = {k: f"{stats[k]} {units[k]}" for k in stats}
                communication_volume[idx] = ', '.join(
                    "{!s}:{!r}".format(key, val)
                    for (key, val) in newd.items())
                f_times[idx].append(f_time)
                b_times[idx].append(b_time)
            else:
                parts.append(idx)

    # calculate mean and variance
    return mean_var(f_times), mean_var(b_times), communication_volume


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


def cuda_time(partition, inputs, recomputation=True):
    # now we move partition to GPU
    partition = partition.to('cuda')
    partition.device = 'cuda'
    b_time, outputs = cuda_backward(partition,
                                    inputs,
                                    recomputation=recomputation)
    f_time = cuda_forward(partition, inputs, recomputation=recomputation)
    partition = partition.cpu()
    partition.device = 'cpu'
    return f_time, b_time, outputs


def cuda_backward(partition, inputs, recomputation=True):
    ''' measure forward/backward time of a partition on the GPU
    '''
    # now we move inputs to GPU
    inputs = [i.to('cuda') for i in inputs]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device='cuda')
    start.record()
    outputs = partition(*inputs)
    if not recomputation:
        start = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device='cuda')
        start.record()
    loss = sum(o.norm() for o in outputs)
    loss.backward()
    end.record()
    torch.cuda.synchronize(device='cuda')
    b_time = (start.elapsed_time(end))

    return b_time, outputs


def cuda_forward(partition, inputs, recomputation=True):
    # now we move inputs to GPU
    inputs = [i.to('cuda') for i in inputs]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device='cuda')
    with torch.set_grad_enabled(not recomputation):
        start.record()
        partition(*inputs)
        end.record()
        torch.cuda.synchronize(device='cuda')
        f_time = (start.elapsed_time(end))
    return f_time


def cpu_time(partition, inputs, recomputation=True):
    ''' measure forward/backward time of a partition on the CPU
    '''
    partition = partition.to('cpu')
    partition.device = 'cpu'
    b_time, outputs = cpu_backward(partition,
                                   inputs,
                                   recomputation=recomputation)
    f_time = cpu_forward(partition, inputs, recomputation=recomputation)

    return f_time, b_time, outputs


def cpu_forward(partition, inputs, recomputation=True):
    inputs = [i.cpu() for i in inputs]
    with torch.set_grad_enabled(not recomputation):
        start = time.time()
        partition(*inputs)
        end = time.time()
        f_time = 1000 * (end - start)

    return f_time


def cpu_backward(partition, inputs, recomputation=True):
    inputs = [i.cpu() for i in inputs]
    start = time.time()
    outputs = partition(*inputs)
    if not recomputation:
        start = time.time()
    loss = sum(o.norm() for o in outputs)
    loss.backward()
    end = time.time()
    b_time = 1000 * (end - start)

    return b_time, outputs


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


def theoretical_analysis(graph, partition_config, recomputation=True):
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
        outputs = [nodes[name] for name in partition_config[i]['outputs']]
        cache = dict()
        parallel_f[i] = 0
        parallel_b[i] = 0
        for o in outputs:
            f, b = parallel_execution_analysis(o, i, cache)
            parallel_f[i] = max(parallel_f[i], f)
            parallel_b[i] = max(parallel_b[i], b)

        if recomputation:
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
# imbalance computation
# ##################################


def worst_imbalance(times):
    return min(times.values()) / max(times.values())


def topology_aware_imbalance(f_times, b_times, cutting_edges):
    ''' find the lowest balance between 2 connected partitions
    '''
    f_imbalance = b_imbalance = 10
    for u, v in cutting_edges:
        f_ratio = min(f_times[u.part], f_times[v.part]) / \
            max(f_times[u.part], f_times[v.part])

        b_ratio = min(b_times[u.part], b_times[v.part]) / \
            max(b_times[u.part], b_times[v.part])

        if f_ratio < f_imbalance:
            f_imbalance = f_ratio

        if b_ratio < b_imbalance:
            b_imbalance = b_ratio

    return f_imbalance, b_imbalance


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
