import warnings
from typing import Dict

import numpy as np

from autopipe.autopipe.model_profiling import Graph, Node
from autopipe.autopipe.utils import FullExecTimes


def partition_and_match_weights_until_last_partition_is_with_no_recomputation(graph: Graph,
                                                                              weights: Dict[Node, FullExecTimes],
                                                                              partitioning_method,
                                                                              partition_profiled_graph_fn,
                                                                              n_runs_limit=20):
    print("-I- partition_and_match_weights_until_last_partition_is_with_no_recomputation")
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

        current_mistakes, d, generated_last_stage_scopes, graph = partition_and_check(graph,
                                                                                      last_partition_scopes,
                                                                                      partition_profiled_graph_fn,
                                                                                      weights)

        history[n_runs] = dict(last_partition_scopes=last_partition_scopes,
                               generated_last_stage_scopes=generated_last_stage_scopes,
                               d=d
                               )
        # set current scopes as model scopes
        last_partition_scopes = generated_last_stage_scopes

        # log something
        print(f"run:{n_runs}", d)

    if not (current_mistakes > allowed_mistakes):
        print(f"Success! got {current_mistakes} mistakes after {n_runs} runs")
    elif not (n_runs_limit < 0 or n_runs < n_runs_limit):
        print(f"Breaking after reaching run limit of {n_runs_limit}!")
        current_mistakes, graph, mistakes_min = restore_best_from_history(graph, history,
                                                                          partition_profiled_graph_fn, weights)

        if current_mistakes != mistakes_min:
            warnings.warn(f"current_mistakes != mistakes_min, {current_mistakes} != {mistakes_min}")

        if current_mistakes > 2:
            graph = exhustive_search_for_last_partition(graph, history, n_runs, partition_profiled_graph_fn, weights)
    return graph


def exhustive_search_for_last_partition(graph, history, n_runs, partition_profiled_graph_fn, weights):
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
        current_mistakes, d, generated_last_stage_scopes, graph = partition_and_check(graph,
                                                                                      last_partition_scopes,
                                                                                      partition_profiled_graph_fn,
                                                                                      weights)

        exhaustive_search_history[n_runs] = dict(last_partition_scopes=last_partition_scopes,
                                                 generated_last_stage_scopes=generated_last_stage_scopes,
                                                 d=d
                                                 )
        print(f"final_countdown_iteration:{i}/{len(topo_sorted_scopes)}", d)
    current_mistakes, graph, mistakes_min = restore_best_from_history(graph, exhaustive_search_history,
                                                                      partition_profiled_graph_fn, weights)
    return graph


def restore_best_from_history(graph, history, partition_profiled_graph_fn, weights):
    i_min = list(history.keys())[int(np.argmin([v['d']['mistakes'] for v in history.values()]))]
    mistakes_min = history[i_min]['d']['mistakes']
    print([history[i]['d']['mistakes'] for i in history])
    print(f"Restoring best point in history")
    print(f"Taking best seen: {mistakes_min} mistakes after {i_min} runs")
    # restore the best point from  history
    last_partition_scopes = history[i_min]['last_partition_scopes']
    current_mistakes, d, generated_last_stage_scopes, graph = partition_and_check(graph,
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
