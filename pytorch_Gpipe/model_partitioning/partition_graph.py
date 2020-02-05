
from typing import Optional, Dict, List

from ..model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction
from .process_partition import post_process_partition

__all__ = ["METIS_partition"]


def METIS_partition(graph: Graph, num_partitions: int,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    **METIS_opts: Dict) -> Graph:
    import nxmetis

    G = graph.asNetworkx(directed=False,
                         node_weight_function=node_weight_function,
                         edge_weight_function=edge_weight_function)

    options = nxmetis.MetisOptions(**METIS_opts)
    objval, parts = nxmetis.partition(G, num_partitions, options=options)
    parts = sorted((idx, n) for n, p in enumerate(parts)for idx in p)
    parts = [n for _, n in parts]

    for node, part in zip(graph.nodes, parts):
        node.part = part
    n_parts = set(parts)
    post_process_partition(graph)

    actual_nparts = len({n.part for n in graph.nodes})

    if(actual_nparts < num_partitions):
        print(
            f"expected {num_partitions} partitions but only {actual_nparts} found implicating that the model to partition is too small")
        print("consider increasing the depth of graph or disabling the basic blocks option")
        print(f"before post processing there were {n_parts} partitions")
    return graph
