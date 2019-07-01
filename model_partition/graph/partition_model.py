from .control_flow_graph import build_graph_from_trace
from .process_partition import post_process_partition
from .optimize_graph import optimize_graph
from .METIS_graph_partition import part_graph


def partition_model(model, num_gpus, *sample_batch, max_depth=100, basic_blocks=None, weights=None):

    if weights is None:
        weights = {}

    graph = build_graph_from_trace(
        model, *sample_batch, max_depth=max_depth, weights=weights, basic_block=basic_blocks)

    optimize_graph(graph)

    adjlist = graph.adjacency_list()
    nodew = graph.get_weights()

    assert(len(adjlist) == len(nodew))

    weights = [weight_func(w) for w in nodew]

    edge_cut, partition = part_graph(
        adjlist, nparts=num_gpus, algorithm="metis", nodew=weights, contig=1)

    post_process_partition(graph, partition)
    return graph, partition, edge_cut


# TODO decide on weighting functional
def weight_func(w):
    if isinstance(w, tuple):
        return int(100*(w.forward_time+w.backward_time)/2)
    return 0
