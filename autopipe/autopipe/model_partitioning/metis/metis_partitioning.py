from collections import defaultdict
from typing import Optional, Dict

from .post_process import post_process_partition
from ..mixed_pipe.partition_mixed_pipe import stages_from_bins, convert_handle_missing_print
from ...model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction


def metis_partition(graph: Graph,
                    num_partitions: int,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_only_graph: bool = True,
                    use_virtual_stages: bool = True,
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

    if use_virtual_stages:
        graph.topo_sort()

    if use_layers_only_graph:
        layers_graph, layers_to_original = graph.layers_graph()
        G = layers_graph
    else:
        G = graph

    work_graph = G

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

        if not use_virtual_stages:
            for node, stage_id in zip(work_graph.nodes, parts):
                node.stage_id = stage_id
        else:
            # Virtual stages
            unique_gpu_ids = set()
            bins = defaultdict(list)

            for node, gpu_id in zip(work_graph.nodes, parts):
                if node in work_graph.inputs:
                    continue
                node.gpu_id = gpu_id
                unique_gpu_ids.add(gpu_id)
                bins[gpu_id].append(node)

            #### Taken from 2dbin
            nodes = [n for n in work_graph.nodes if n not in work_graph.inputs]
            id_to_node = {node.id: node for node in nodes}

            stages_from_bins(graph=work_graph, bins=bins, id_to_node_worked_on=id_to_node)

        try:
            post_process_partition(work_graph, edge_weight_function, assert_output_types=False,
                                   verbose_on_error=verbose_on_error)
            fail = False
            break
        except (Exception, RuntimeError, AssertionError) as e:
            last_exception = e

    if fail:
        print(f"-I- METIS could not find a valid partitioning")
        raise last_exception

    # TODO this was written without virtual stages.
    n_parts = set(parts)
    actual_nparts = len({n.stage_id for n in work_graph.nodes})

    if (actual_nparts < num_partitions):
        print("This is deprecated....")
        print(
            f"-I- expected {num_partitions} partitions but only {actual_nparts} found"
            " implicating that the model to partition is too small")
        print(
            "consider increasing the depth of graph or disabling the basic blocks option"
        )
        print(f"before post processing there were {n_parts} partitions")

    if use_layers_only_graph:
        graph.induce_layer_partition(work_graph, layers_to_original)

    if use_virtual_stages:
        stage_to_gpu_map = convert_handle_missing_print(bins=bins, graph=graph, verbose=False)

    return graph
