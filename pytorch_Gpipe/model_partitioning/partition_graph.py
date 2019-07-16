
from ..METIS import METIS_partition
from .process_partition import post_process_partition
from ..model_profiling import Graph
from typing import Optional, Callable, Any, List

__all__ = ["partition_graph"]


def partition_graph(graph: Graph, num_partitions: int, weighting_function: Optional[Callable[[Any], int]] = None, **METIS_opts) -> Graph:
    '''
    partition the graph using METIS's PartGraphKway and then optimizes it to our needs

    Parameters
    ----------
    graph:
        the Graph object to partition
    num_partitions:
        the requested number of partitions
    weighting_function:
        a weighting function that transforms the graph weights to non negative integers
        if not specified a default function will be used
    METIS_opts:
        additional options to pass to METIS
        for eg. for the option METIS_OPTION_SEED pass seed=value
    '''

    wfunc = weighting_function if weighting_function != None else default_weight_func

    adjlist = graph.adjacency_list()
    nodew = graph.get_weights()

    assert(len(adjlist) == len(nodew))

    weights = [wfunc(w) for w in nodew]

    if 'seed' not in METIS_opts:
        METIS_opts['seed'] = 0

    if 'contig' not in METIS_opts:
        METIS_opts['contig'] = 1

    partition, _ = METIS_partition(adjlist, nparts=num_partitions, algorithm="metis",
                                   nodew=weights, **METIS_opts)

    post_process_partition(graph, partition)

    actual_nparts = len({n.part for n in graph.nodes})

    if(actual_nparts < num_partitions):
        print(
            f"expected {num_partitions} partitions but only {actual_nparts} found implicating that the model to partition is too small")
        print("consider increasing the depth of graph or disabling the basic blocks option")
    return graph


def default_weight_func(w):
    if hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
        return max(int(100*(w.forward_time+w.backward_time)/2), 1)
    return 1
