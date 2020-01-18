
from typing import Callable, Optional, Dict, Union

from ..model_profiling import Graph, Profile
from .process_partition import post_process_partition
import networkx as nx
import nxmetis

__all__ = ["partiton_graph"]


def partiton_graph(graph: Graph, num_partitions: int, weighting_function: Callable[[Union[Profile, int]], int], **METIS_opts: Dict) -> Graph:
    weights = {node.idx: weighting_function(node.weight)
               for node in graph.nodes}

    G = graph.asNetworkx()
    nx.set_node_attributes(G, weights, 'weight')
    options = nxmetis.MetisOptions(**METIS_opts)
    objval, parts = nxmetis.partition(G, num_partitions, options=options)
    parts = sorted((idx, n) for n, p in enumerate(parts)for idx in p)
    parts = [n for _, n in parts]

    post_process_partition(graph, parts)

    actual_nparts = len({n.part for n in graph.nodes})

    if(actual_nparts < num_partitions):
        print(
            f"expected {num_partitions} partitions but only {actual_nparts} found implicating that the model to partition is too small")
        print("consider increasing the depth of graph or disabling the basic blocks option")
    return graph
