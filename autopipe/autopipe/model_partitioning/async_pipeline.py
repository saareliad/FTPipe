"""Meta algorithm to partition heterogeneous stages
In some work schedulers there are two types of stages in the pipeline:
(A) the last stage, which does not recompute (B) other stages, which do recompute.
Without further actions, the execution of the last stage would not match its profiling (it would be faster by approximately 33)
and result in underutilization.
we accommodates by matching the correct profiles for each block.

the process is as follows:
(1)	For each basic block: profile it twice:
with Recomputation and without Recomputation.
(2)	Hold for each block a Boolean value, indicating which profile to use (default: without Recomputation)
(3)	Run partitioning: during partitioning, each block uses only the profile it was mapped to, according to its Boolean value.
(4)	Count mistakes, i.e., blocks which used the wrong profile.
(5). If there are mistakes, change Boolean values to match the last result, then run (3) again.

If (5) still finds mistakes after several trials, we break and run an exhaustive search over the subset of blocks mapped to the last stage during the 1st  run of (3) and take the matching with minimal mistakes.

Note that the identity of block-to-stage mapping is unknown at profiling.  However,
some partitioning algorithms can switch profiles dynamically,  sparing the meta-algorithm. Nevertheless, we find the meta-algorithm to have only a small overhead and work for more partitioning schemes.
"""
import warnings
from typing import Dict

import numpy as np

from autopipe.autopipe.model_profiling import Graph, Node
from autopipe.autopipe.utils import FullExecTimes


def partition_and_match_weights_until_last_partition_is_with_no_recomputation(graph: Graph,
                                                                              weights: Dict[Node, FullExecTimes],
                                                                              partitioning_method,
                                                                              partition_profiled_graph_fn,
                                                                              n_runs_limit=10,
                                                                              do_exhustive_search_for_last_partition=True,
                                                                              max_memory_usage_r=None, max_memory_usage_nr=None):
    print("-I- partition_and_match_weights_until_last_partition_is_with_no_recomputation")

    warnings.warn("need to set max memory usage: currently doing this only for recomputation.")  # TODO:
    if max_memory_usage_r:
        for node in graph.nodes:
            if node.scope in max_memory_usage_r:
                node.max_memory_bytes = max_memory_usage_r[node.scope]

    # protect graph:
    saved_state = graph.state()
    allowed_mistakes = 0
    # HACK: allow mistakes for multilevel and acyclic...
    if partitioning_method == "ACYCLIC":
        allowed_mistakes += 2

    last_partition_scopes = set()
    current_mistakes = allowed_mistakes + 1
    n_runs = 0

    history = dict()
    while current_mistakes > allowed_mistakes and (n_runs_limit < 0 or n_runs < n_runs_limit):
        n_runs += 1

        current_mistakes, d, generated_last_stage_scopes, graph = partition_and_check(Graph.from_state(saved_state),
                                                                                      last_partition_scopes,
                                                                                      partition_profiled_graph_fn,
                                                                                      weights)

        history[n_runs] = dict(last_partition_scopes=last_partition_scopes,
                               generated_last_stage_scopes=generated_last_stage_scopes,
                               d=d,
                               graph_state=graph.state()
                               )
        # set current scopes as model scopes
        last_partition_scopes = generated_last_stage_scopes

        # log something
        print(f"run:{n_runs}", d)

    if not (current_mistakes > allowed_mistakes):
        print(f"Success! got {current_mistakes} mistakes after {n_runs} runs")
    elif not (n_runs_limit < 0 or n_runs < n_runs_limit):
        print(f"Breaking after reaching run limit of {n_runs_limit}!")
        current_mistakes, graph, mistakes_min = restore_best_from_history(saved_state, history,
                                                                          partition_profiled_graph_fn, weights)

        if current_mistakes != mistakes_min:
            warnings.warn(f"current_mistakes != mistakes_min, {current_mistakes} != {mistakes_min}")

        if current_mistakes > 2 and do_exhustive_search_for_last_partition:
            graph = exhustive_search_for_last_partition(saved_state, graph, history, n_runs, partition_profiled_graph_fn, weights, smallest_fp_with_zero_fp=True)
    return graph


def exhustive_search_for_last_partition(saved_state, graph, history, n_runs, partition_profiled_graph_fn, weights, smallest_fp_with_zero_fp=False):

    if smallest_fp_with_zero_fp:
        cands = []
        for i,v in history.items():
            d = v['d']
            if d['fp'] > 0:
                continue
            cands.append((i, (d['fn'], -d['correct'])))

        best = cands[0]
        for c in cands[1:]:
            if c[1] < best:
                best = c

        possible_scopes = set(history[best[0]]['generated_last_stage_scopes'])
    else:
        # Taking the first point in history
        possible_scopes = set(history[1]['generated_last_stage_scopes'])
    # topo sort scopes
    scope_to_id = {}
    for n in graph.nodes:
        if n.scope in possible_scopes:
            scope_to_id[n.scope] = n.id
    topo_sorted_scopes = sorted(possible_scopes, key=lambda x: scope_to_id[x])
    print("Guessing prev-option didn't converge,")
    print("Doing exhaustive search over last stage IDs and taking best fit")
    exhaustive_search_history = dict()
    # TODO: can skip some options by last param in range() call
    for i in range(len(topo_sorted_scopes)):
        last_partition_scopes = topo_sorted_scopes[i:]
        current_mistakes, d, generated_last_stage_scopes, graph = partition_and_check(Graph.from_state(saved_state),
                                                                                      last_partition_scopes,
                                                                                      partition_profiled_graph_fn,
                                                                                      weights)

        exhaustive_search_history[i] = dict(last_partition_scopes=last_partition_scopes,
                                                 generated_last_stage_scopes=generated_last_stage_scopes,
                                                 d=d,
                                                graph_state=graph.state()
                                                 )
        print(f"final_countdown_iteration:{i}/{len(topo_sorted_scopes)}", d)
    current_mistakes, graph, mistakes_min = restore_best_from_history(saved_state, exhaustive_search_history,
                                                                      partition_profiled_graph_fn, weights)
    return graph


def restore_best_from_history(saved_state, history, partition_profiled_graph_fn, weights):
    # saved_state is initial saved_state
    i_min = list(history.keys())[int(np.argmin([v['d']['mistakes'] for v in history.values()]))]
    mistakes_min = history[i_min]['d']['mistakes']
    print([history[i]['d']['mistakes'] for i in history])
    print(f"Restoring best point in history")
    print(f"Taking best seen: {mistakes_min} mistakes after {i_min} runs")
    # restore the best point from  history
    min_hist = history[i_min]
    if 'graph_state' in min_hist:
        graph_state = min_hist['graph_state']
        current_mistakes = mistakes_min
        graph = Graph.from_state(graph_state)
    else:
        print("Partitioning again to restore history")
        warnings.warn("must start from clear state!")
        last_partition_scopes = history[i_min]['last_partition_scopes']
        current_mistakes, d, generated_last_stage_scopes, graph = partition_and_check(Graph.from_state(saved_state),
                                                                                      last_partition_scopes,
                                                                                      partition_profiled_graph_fn,
                                                                                      weights)
    return current_mistakes, graph, mistakes_min


def partition_and_check(graph, last_partition_scopes, partition_profiled_graph_fn, weights):
    for n in graph.nodes:
        if n.scope in last_partition_scopes:
            n.weight = weights[n.id].no_recomputation
        else:
            n.weight = weights[n.id].recomputation
    # Partition
    graph = partition_profiled_graph_fn(graph)
    # Load last partition last stage scopes
    last_p = max((n.stage_id for n in graph.nodes))
    generated_last_stage_scopes = [
        n.scope for n in graph.nodes if n.stage_id == last_p
    ]
    # Count mistakes (false positives and false negatives)
    A = set(last_partition_scopes)
    B = set(generated_last_stage_scopes)
    intersection = A & B
    correct = len(intersection)
    fp = len(A) - correct  # we predicted: true, result: false
    fn = len(B) - correct  # we predicted: false, result: true
    current_mistakes = fp + fn
    # stats:
    d = dict(correct=correct, fp=fp, fn=fn, mistakes=current_mistakes)
    return current_mistakes, d, generated_last_stage_scopes, graph
