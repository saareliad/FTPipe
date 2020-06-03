from typing import Optional, Dict, List

from ...model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction
from .process_partition import post_process_partition

__all__ = ["METIS_partition"]


def METIS_partition(graph: Graph,
                    num_partitions: int,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_only_graph: bool = True,
                    **METIS_opts: Dict) -> Graph:
    '''
    performs METIS Kway partitioning on the given graph

    Parameters:
    graph:
        the Graph to partition
    num_partitions:
        the number of partitions
    node_weight_function:
        an optional weight function for the nodes should be a function from Node to int
        if not given a default weight of 1 will be given to all nodes
    edge_weight_function:
        an optional weight function for the edges should be a function (Node,Node) to int
        if not given a default value of 1 will be given to all edges
    use_layers_only_graph:
        whether to partition a smaller version of the graph containing only the layers
        (usefull fo big models with lots of unprofiled ops)
    METIS_opts:
        additional kwargs to pass to the METIS partitioning algorithm
    '''
    import nxmetis

    if use_layers_only_graph:
        layers_graph, layers_to_original = graph.layers_graph()
        G = layers_graph
    else:
        G = graph

    G = G.asNetworkx(directed=False,
                     node_weight_function=node_weight_function,
                     edge_weight_function=edge_weight_function)

    options = nxmetis.MetisOptions(**METIS_opts)
    objval, parts = nxmetis.partition(G,
                                      num_partitions,
                                      options=options,
                                      node_weight='weight',
                                      node_size='size',
                                      edge_weight='weight',
                                      recursive=False)
    parts = sorted((idx, n) for n, p in enumerate(parts) for idx in p)
    parts = [n for _, n in parts]

    if use_layers_only_graph:
        for node, part in zip(layers_graph.nodes, parts):
            node.part = part
        induce_layer_partition(graph, layers_graph, layers_to_original)
    else:
        for node, part in zip(graph.nodes, parts):
            node.part = part

    n_parts = set(parts)
    post_process_partition(graph,edge_weight_function)

    actual_nparts = len({n.part for n in graph.nodes})

    if (actual_nparts < num_partitions):
        print(
            f"expected {num_partitions} partitions but only {actual_nparts} found"
            " implicating that the model to partition is too small")
        print(
            "consider increasing the depth of graph or disabling the basic blocks option"
        )
        print(f"before post processing there were {n_parts} partitions")
    return graph


def induce_layer_partition(original_graph: Graph,layers_graph:Graph,
                           layers_to_original: Dict[int, int]) -> Graph:
    old_to_new = {v: k for k, v in layers_to_original.items()}
    N = len(original_graph)
    #iterate in reverse order
    for idx in range(len(original_graph.nodes)):
        node = original_graph[N-idx-1]
        if node.id in old_to_new:
            node.part = layers_graph[old_to_new[node.id]].part
        else:
            # as we iterate in reverse topological order we've already handled this node's outupts
            #select the lowest partition index to ensure no cycles are created
            node.part = sorted(node.out_edges,key=lambda n: n.part)[0].part
        assert node.part >= 0

    return original_graph
