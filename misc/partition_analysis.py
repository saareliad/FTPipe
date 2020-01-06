import torch
from collections import deque
import time
from contextlib import nullcontext


def run_analysis(sample, graph, config, n_iter, recomputation=True, bandwidth_gps=16):
    # thoeretical analysis
    sequential_f, sequential_b, parallel_f, parallel_b = theoretical_analysis(graph, config,
                                                                              recomputation=recomputation)
    edges = edge_cut(graph)
    # theoretical analysis based on the graph assuming the computation is sequential
    theoretical_sequential_b_imbalance = worst_imbalance(sequential_b)
    theoretical_sequential_f_imbalance = worst_imbalance(sequential_f)
    topology_aware_sequential_f_imbalance, topology_aware_sequential_b_imbalance = topology_aware_imbalance(sequential_f,
                                                                                                            sequential_b, edges)
    # theoretical anaysis based on the graph assuming the computation is fully parallel
    theoretical_parallel_b_imbalance = worst_imbalance(parallel_b)
    theoretical_parallel_f_imbalance = worst_imbalance(parallel_f)
    topology_aware_parallel_f_imbalance, topology_aware_parallel_b_imbalance = topology_aware_imbalance(parallel_f,

                                                                                                        parallel_b, edges)
    # real statistics based on generated partitions
    real_f_times, real_b_times, comm_volume = profile_execution(sample, config, n_iter,
                                                                recomputation=recomputation, bandwidth_gps=bandwidth_gps)
    real_b_imbalance = worst_imbalance(real_b_times)
    real_f_imbalance = worst_imbalance(real_f_times)
    topology_aware_real_f_imbalance, topology_aware_real_b_imbalance = topology_aware_imbalance(real_f_times,
                                                                                                real_b_times, edges)

    print("cutting edges are edges between partitions")
    print(f"number of cutting edges: {len(edges)}\n")

    if recomputation:
        print("backward times include recomputation")
    else:
        print("backward times do not include recomputation")

    print(
        f"\ntheoretical times are execution time based on sum of graph weights ms\nsequential forward {sequential_f}\nsequential backward {sequential_b}")
    print(f"parallel forward {parallel_f}\nparallel backward {parallel_b}")

    print(
        f"\nreal times are based on real measurements of execution time of generated partitions ms\nforward {real_f_times}\nbackward {real_b_times}")

    print("\nimbalance is ratio of computation time between fastest and slowest parts between 0 and 1 higher is better")
    print(
        f"theoretical sequential imbalance:\nforward {theoretical_sequential_f_imbalance}\nbackward {theoretical_sequential_b_imbalance}")
    print(
        f"theoretical parallel imbalance:\nforward {theoretical_parallel_f_imbalance}\nbackward {theoretical_parallel_b_imbalance}")
    print(
        f"\nreal imbalance:\nforward {real_f_imbalance}\nbackward {real_b_imbalance}")

    print("\ntopology aware imbalance is worst imbalance between 2 connected partitions")
    print(
        f"theoretical sequential topology aware imbalance:\nforwad {topology_aware_sequential_f_imbalance}\nbackward {topology_aware_sequential_b_imbalance}")
    print(
        f"theoretical parallel topology aware imbalance:\nforwad {topology_aware_parallel_f_imbalance}\nbackward {topology_aware_parallel_b_imbalance}")
    print(
        f"\nreal topology aware imbalance:\nforwad {topology_aware_real_f_imbalance}\nbackward {topology_aware_real_b_imbalance}")

    print(
        f"\ncommunication volumes size of activations of each partition")

    for idx, volume in comm_volume.items():
        print(f"{idx}: {volume}")

    # latency stuff not in use right now
    # forward_dependencies, backward_dependencies = find_dependencies(edges,
    #                                                                 len(theoretical_f_times))
    # ideal_t_f_latency = calculate_ideal_latency(forward_dependencies,
    #                                             theoretical_f_times)
    # ideal_t_b_latency = calculate_ideal_latency(backward_dependencies,
    #                                             theoretical_b_times)
    # ideal_sum_t_f_latency = calculate_ideal_latency(forward_dependencies,
    #                                                 f_sum)
    # ideal_sum_t_b_latency = calculate_ideal_latency(backward_dependencies,
    #                                                 b_sum)
    # ideal_a_f_latency = calculate_ideal_latency(forward_dependencies,
    #                                             real_f_times)
    # ideal_a_b_latency = calculate_ideal_latency(backward_dependencies,
    #                                             real_b_times)
    # print(
    #     f"\nideal latency is the time that passes for a forward/backward pass to reach and leave the partition")
    # print(
    #     f"ideal theoretical latencies ms\nforward {ideal_t_f_latency}\nbackward {ideal_t_b_latency}")
    # print(
    #     f"\nideal real latencies ms\nforward {ideal_a_f_latency}\nbackward {ideal_a_b_latency}")


def profile_execution(model_inputs, partition_config, n, recomputation=True, bandwidth_gps=16):
    '''perfrom forward/backward passes and measure execution times accross n batches
    '''
    n_partitions = sum([1 for k in partition_config if isinstance(k, int)])
    f_times = {i: 0 for i in range(n_partitions)}
    b_times = {i: 0 for i in range(n_partitions)}

    communication_volume = {}
    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs,)

    for _ in range(n):
        parts = deque(range(n_partitions))
        activations = {}
        for i, t in zip(partition_config['model inputs'], model_inputs):
            # save activations on CPU in order to save GPU memory
            activations[i] = t.cpu()

        # perform one run of the partitions
        while len(parts) > 0:
            idx = parts.popleft()
            if all(tensor in activations for tensor in partition_config[idx]['inputs']):
                inputs = [activations[tensor]
                          for tensor in partition_config[idx]['inputs']]

                # input statistics
                in_size_mb = 0
                recv_time = 0
                for t in inputs:
                    t_mb = (t.nelement() * t.element_size()) / 1e6
                    t_recv = (t_mb/(bandwidth_gps*1e3))
                    in_size_mb += t_mb
                    recv_time = max(recv_time, t_recv)
                recv_time *= 1e3

                # time measurement
                if torch.cuda.is_available():
                    f_time, b_time, outputs = cuda_time(partition_config[idx]['model'],
                                                        inputs, recomputation=recomputation)
                else:
                    f_time, b_time, outputs = cpu_time(partition_config[idx]['model'],
                                                       inputs, recomputation=recomputation)

                # output statistics
                out_size_mb = 0
                send_time = 0
                for o, t in zip(partition_config[idx]['outputs'], outputs):
                    # save activation on CPU in order to save GPU memory
                    activations[o] = t.cpu()
                    t_mb = (t.nelement() * t.element_size()) / 1e6
                    t_send = (t_mb/(bandwidth_gps*1e3))
                    out_size_mb += t_mb
                    send_time = max(t_send, send_time)

                send_time *= 1e3
                communication_volume[idx] = f"input size: {in_size_mb} MB recieve_time: {recv_time} ms out: {out_size_mb} MB send time: {send_time} ms"
                f_times[idx] += f_time
                b_times[idx] += b_time
            else:
                parts.append(idx)

    avg_f_times = {i: v/n for i, v in f_times.items()}
    avg_b_times = {i: v/n for i, v in b_times.items()}

    return avg_f_times, avg_b_times, communication_volume


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


def cuda_time(partition, inputs, recomputation=True):
    # now we move partition to GPU
    partition = partition.to('cuda')
    partition.device = 'cuda'
    b_time, outputs = cuda_backward(partition, inputs,
                                    recomputation=recomputation)
    f_time = cuda_forward(partition, inputs, recomputation=recomputation)
    partition = partition.cpu()
    partition.device = 'cpu'
    return f_time, b_time, outputs


def cuda_backward(partition, inputs, recomputation=True):
    ''' measure forward/backward time of a partition on the GPU
    '''
    # now we move inputs to GPU
    inputs = [i.to('cuda').detach_() for i in inputs]
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
    inputs = [i.to('cuda').detach_() for i in inputs]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device='cuda')
    with torch.no_grad() if recomputation else nullcontext():
        start.record()
        outputs = partition(*inputs)
        end.record()
        torch.cuda.synchronize(device='cuda')
        f_time = (start.elapsed_time(end))

        return f_time


def cpu_time(partition, inputs, recomputation=True):
    ''' measure forward/backward time of a partition on the CPU
    '''
    partition = partition.to('cpu')
    partition.device = 'cpu'
    b_time, outputs = cpu_backward(partition, inputs,
                                   recomputation=recomputation)
    f_time = cpu_forward(partition, inputs, recomputation=recomputation)

    return f_time, b_time, outputs


def cpu_forward(partition, inputs, recomputation=True):
    inputs = [i.cpu().detach_() for i in inputs]
    with torch.no_grad() if recomputation else nullcontext():
        start = time.time()
        outputs = partition(*inputs)
        end = time.time()
        f_time = 1000 * (end - start)

        return f_time


def cpu_backward(partition, inputs, recomputation=True):
    inputs = [i.cpu().detach_() for i in inputs]
    start = time.time()
    outputs = partition(*inputs)
    if not recomputation:
        start = time.time()
    loss = sum(o.norm() for o in outputs)
    loss.backward()
    end = time.time()
    b_time = 1000 * (end - start)

    return b_time, outputs


def extract_time(w, forward=False):
    if hasattr(w, "weight"):
        w = w.weight
    if not hasattr(w, "forward_time"):
        return 0
    if forward:
        return w.forward_time
    return w.backward_time


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


def run_partitions(model_inputs, partition_config):
    n_partitions = sum([1 for k in partition_config if isinstance(k, int)])

    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs,)

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
        if all(tensor in activations for tensor in partition_config[idx]['inputs']):
            inputs = [activations[tensor]
                      for tensor in partition_config[idx]['inputs']]
            outs = partition_config[idx]['model'](*inputs)
            for o, t in zip(partition_config[idx]['outputs'], outs):
                activations[o] = t
        else:
            parts.append(idx)

    return [activations[o] for o in partition_config['model outputs']]


# latency stuff not in use
def calculate_ideal_latency(dependencies, times):
    '''calculates latency as sum of exec time of partition dependencies
    '''
    n_parts = len(times)

    ideal = {i: times[i]+sum(times[j] for j in dependencies[i])
             for i in range(n_parts)}

    return ideal


def find_dependencies(cutting_edges, n_parts):
    ''' find input/output dependencies between all partitions
        a partiton is dependent on another if there is a path between them in the graph
    '''
    forward_dependencies = {i: set() for i in range(n_parts)}
    backward_dependencies = {i: set() for i in range(n_parts)}
    while True:
        changed = False
        for u, v in cutting_edges:
            # update input paths
            prev = len(forward_dependencies[v.part])
            forward_dependencies[v.part].add(u.part)
            forward_dependencies[v.part].update(forward_dependencies[u.part])
            if len(forward_dependencies[v.part]) > prev:
                changed = True

            # update output paths:
            prev = len(backward_dependencies[u.part])
            backward_dependencies[u.part].add(v.part)
            backward_dependencies[u.part].update(backward_dependencies[v.part])
            if len(backward_dependencies[u.part]) > prev:
                changed = True

        if not changed:
            break

    return forward_dependencies, backward_dependencies
