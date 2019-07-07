
from ..METIS import METIS_partition
from .process_partition import post_process_partition
from ..model_profiling import Graph
from typing import Optional, Callable, Any


def partition_graph(graph: Graph, num_partitions: int, weighting_function: Optional[Callable[[Any], int]] = None):
    wfunc = weighting_function if weighting_function != None else weight_func

    adjlist = graph.adjacency_list()
    nodew = graph.get_weights()

    assert(len(adjlist) == len(nodew))

    weights = [wfunc(w) for w in nodew]

    edge_cut, partition = METIS_partition(adjlist, nparts=num_partitions, algorithm="metis",
                                          nodew=weights, contig=1)

    post_process_partition(graph, partition)

    actual_nparts = len({n.part for n in graph.nodes})

    if(actual_nparts < num_partitions):
        print(
            f"expected {num_partitions} partitions but only {actual_nparts} found implicating that the model to partition is too small")
        print("consider increasing the depth of graph or disabling the basic blocks option")
    return graph, partition, edge_cut


def weight_func(w):
    if isinstance(w, tuple) and hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
        return int(100*(w.forward_time+w.backward_time)/2)
    return 0
