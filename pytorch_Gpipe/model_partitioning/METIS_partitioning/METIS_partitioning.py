from typing import Optional, Dict

from .process_partition import post_process_partition
from ...model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction

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

    attempts = METIS_opts.pop("attempts", 1)
    verbose_on_error = METIS_opts.pop("verbose_on_error", False)
    options = nxmetis.MetisOptions(**METIS_opts)
    fail = True
    last_exception = None

    # METIS partitioning does not enforce an acyclic constraint between partitions
    # which is a must for partitioning computation graphs
    # so we do multiple attempts in the hopes that one of them will give a valid result
    # not that the Acyclic_partitoning does not suffer from this issue (but it can also give an inferior solution)
    for _ in range(attempts):
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
            for node, stage_id in zip(layers_graph.nodes, parts):
                node.stage_id = stage_id
            graph.induce_layer_partition(layers_graph, layers_to_original)
        else:
            for node, stage_id in zip(graph.nodes, parts):
                node.stage_id = stage_id

        try:
            post_process_partition(graph, edge_weight_function, assert_output_types=False,
                                   verbose_on_error=verbose_on_error)
            fail = False
            break
        except (Exception, RuntimeError, AssertionError) as e:
            last_exception = e

    if fail:
        print(f"-I- METIS could not find a valid partitioning")
        raise last_exception

    n_parts = set(parts)
    actual_nparts = len({n.stage_id for n in graph.nodes})

    if (actual_nparts < num_partitions):
        print(
            f"-I- expected {num_partitions} partitions but only {actual_nparts} found"
            " implicating that the model to partition is too small")
        print(
            "consider increasing the depth of graph or disabling the basic blocks option"
        )
        print(f"before post processing there were {n_parts} partitions")
    return graph
