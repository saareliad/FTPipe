
from ..METIS import METIS_partition
from .process_partition import post_process_partition
from ..model_profiling import Graph


def partition_graph(graph: Graph, num_partitions, weighting_function=None):

    wfunc = weighting_function if weighting_function != None else weight_func

    adjlist = graph.adjacency_list()
    nodew = graph.get_weights()

    assert(len(adjlist) == len(nodew))

    weights = [wfunc(w) for w in nodew]

    edge_cut, partition = METIS_partition(adjlist, nparts=num_partitions, algorithm="metis",
                                          nodew=weights, contig=1)

    post_process_partition(graph, partition)
    return graph, partition, edge_cut


# TODO decide on default weighting functiona
def weight_func(w):
    if isinstance(w, tuple) and hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
        return int(100*(w.forward_time+w.backward_time)/2)
    return 0
