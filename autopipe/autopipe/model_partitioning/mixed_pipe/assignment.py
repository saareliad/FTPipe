import warnings
from copy import deepcopy

from tqdm import tqdm

from autopipe.autopipe.model_profiling import Graph
from autopipe.autopipe.model_partitioning.heuristics import NodeMemoryEstimator
from autopipe.autopipe.model_partitioning.mixed_pipe.heap_dict import heapdict
import numpy as np
from math import comb  # python 3.8


def greedy_best_fit(graph: Graph, P, node_weight_function, node_mem_estimator: NodeMemoryEstimator):
    bins = {i: list() for i in range(P)}
    bin_weights = heapdict({i: 0 for i in range(P)})
    bin_memory = heapdict({i: 0 for i in range(P)})

    node_to_weight = {n: node_weight_function(n) for n in graph.non_input_nodes}
    node_to_weight = dict(sorted(node_to_weight.items(), key=lambda item: item[1], reverse=True))

    gpu_mem_threshold_bytes = {i: 9 * 1e9 for i in bins}  # 11 - 512*2 - extra for send recv
    node_to_mem = {n: node_mem_estimator(n) for n in graph.non_input_nodes}

    def check_memory_fit(candidate, bin_id):
        # TODO:  PoC
        if node_to_mem[candidate] + bin_memory[bin_id] > gpu_mem_threshold_bytes[bin_id]:
            print(f"-v- failed to add candidate to GPU {bin_id}")
            return False
        return True

    def choose_bin(node):
        tmp = []
        while bin_weights:
            bin_id, w = bin_weights.peekitem()
            if not check_memory_fit(node, bin_id):
                tmp.append(bin_weights.popitem())
                continue
            # restore - next item may be smaller!
            # it does not really matter, since if we would fail on smallest - we fail on all.
            for i,v in tmp:
                warnings.warn("it is unprobable we got here.")
                bin_weights[i] = v
            return bin_id
        raise RuntimeError("Could not find an assignment which fits memory")

    while node_to_weight:
        node, node_weight = node_to_weight.popitem()
        bin_id = choose_bin(node)
        bins[bin_id].append(node)
        bin_weights[bin_id] += node_weight
        bin_memory[bin_id] += node_to_mem[node]

    return bins




def algorithm_u(ns, m):
    """taken from https://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions
    """
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)
#
# def partition(collection):
#     if len(collection) == 1:
#         yield [collection]
#         return
#
#     first = collection[0]
#     for smaller in partition(collection[1:]):
#         # insert `first` in each of the subpartition's subsets
#         for n, subset in enumerate(smaller):
#             yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
#         # put `first` in its own subset
#         yield [[first]] + smaller

def exhustive_search(graph: Graph, P, node_weight_function, node_mem_estimator: NodeMemoryEstimator, L):

    all_nodes = list(graph.non_input_nodes)
    all_weights = np.array([node_weight_function(x) for x in all_nodes])
    all_mems = np.array([node_mem_estimator(x) for x in all_nodes])
    L_tag = len(all_nodes)

    # gpu_mem_threshold_bytes = {i: 10 * 1e9 for i in bins}
    homogenous_threshold = 10* 1e9

    print(f"Doing exhaustive search ")

    def num_stages(m):
        # TODO:
        raise NotImplementedError()


    best_m_comp = np.inf
    best_solution = None

    for m in tqdm(algorithm_u(list(range(L_tag)), P), desc="exhustive_search"):
        top_mem = max(np.sum(all_mems[b]) for b in m)
        if top_mem > homogenous_threshold:
            continue
        top_comp = max(np.sum(all_weights[b]) for b in m)
        if top_comp < best_m_comp:
            best_m_comp = top_comp
            best_solution = deepcopy(m)

    all_nodes = np.array(all_nodes)
    m = best_solution
    bins = {i: all_nodes[b].tolist() for i,b in enumerate(m)}
    return bins