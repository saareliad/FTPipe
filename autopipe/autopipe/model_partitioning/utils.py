from collections import defaultdict
from typing import Dict, Set, List

from autopipe.autopipe.model_profiling.control_flow_graph import Graph


def re_assign_partition_indices(graph: Graph):
    out_edges = defaultdict(set)
    for node in graph.non_input_nodes:
        for o in node.out_edges:
            out_edges[node.stage_id].add(o.stage_id)
            # handle sinks (why the heck this results in cycles?!)
            # if o.stage_id not in out_edges:
            #     out_edges[node.stage_id] = set()

    # handle sinks: (assuming they are only graph outputs.)
    for node in graph.outputs:
        if node.stage_id not in out_edges:
            out_edges[node.stage_id] = set()

    for i, e in out_edges.items():
        e.discard(i)

    translation = {idx: i for i, idx in enumerate(topological_sort(out_edges))}
    for node in graph.nodes:
        node.stage_id = translation[node.stage_id]


def topological_sort(out_edges: Dict[int, Set[int]]) -> List[int]:
    visited = {i: False for i in out_edges}
    stack = []

    for i in out_edges.keys():
        if not visited[i]:
            _topological_sort(out_edges, i, visited, stack)

    return stack


def _topological_sort(out_edges: Dict[int, Set[int]], v: int, visited: Dict[int, bool], stack: List[int]):
    visited[v] = True

    for i in out_edges[v]:
        if not visited[i]:
            _topological_sort(out_edges, i, visited, stack)

    stack.insert(0, v)


def has_stage_cycles(graph: Graph) -> bool:
    for u in graph.non_input_nodes:
        for v in u.out_edges:
            if v.stage_id < u.stage_id:
                return True

    return False
