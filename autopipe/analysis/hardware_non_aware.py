"""FIXME: DEPRECATED, not accurate, probably incorrect"""
from collections import defaultdict

from autopipe.autopipe.model_profiling import NodeTypes


def theoretical_analysis(graph, recomputation=True, async_pipeline=False):
    """ find execution time of partitions based on the model's graph using 2 a sequential assumption and parallel assumption
        the sequential assumption is that in the partition all operation are linear.
        the parallel assumption assumes that all computation paths are concurrent.
    """
    n_parts = len(set(n.stage_id for n in graph.nodes))
    print(f"Theoretical analysis found n_parts={n_parts}")
    parallel_b = dict()
    parallel_f = dict()

    tensor_names = set()
    stage_outputs = defaultdict(list)
    for n in graph.nodes:
        if (n.type != NodeTypes.IN) and any(o.stage_id != n.stage_id
                                            for o in n.out_edges):
            tensor_names.add(n.scope)
            stage_outputs[n.stage_id].append(n.scope)
        elif n in graph.outputs:
            tensor_names.add(n.scope)
            stage_outputs[n.stage_id].append(n.scope)

    sequential_f = {i: 0 for i in range(n_parts)}
    sequential_b = {i: 0 for i in range(n_parts)}

    nodes = dict()
    for node in graph.nodes:
        # cache relevant nodes to make fetching them faster
        if graph.input_kw_ids.get(node.id, node.scope) in tensor_names:
            nodes[graph.input_kw_ids.get(node.id, node.scope)] = node

        # old way of measuring time as sum of all computation
        sequential_f[node.stage_id] += extract_time(node.weight, forward=True)
        sequential_b[node.stage_id] += extract_time(node.weight, forward=False)

    # new way of measuring time as longest path where all paths are concurrent
    for i in range(n_parts):
        partition_specific_computation = recomputation
        is_last_partition = (i == n_parts - 1)
        if async_pipeline and is_last_partition:
            partition_specific_computation = False

        outputs = [nodes[name] for name in stage_outputs[i]]
        cache = dict()
        parallel_f[i] = 0
        parallel_b[i] = 0
        for o in outputs:
            f, b = parallel_execution_analysis(o, i, cache)
            parallel_f[i] = max(parallel_f[i], f)
            parallel_b[i] = max(parallel_b[i], b)

        if partition_specific_computation:
            sequential_b[i] += sequential_f[i]
            parallel_b[i] += parallel_f[i]

    return sequential_f, sequential_b, parallel_f, parallel_b


def parallel_execution_analysis(node, part_idx, cache):
    # use cache in order to remember common subpaths
    if node.scope in cache:
        return cache[node.scope]
    elif node.stage_id != part_idx:
        cache[node.scope] = (0, 0)
        return 0, 0

    longest_f, longest_b = 0, 0

    for n in node.in_edges:
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
