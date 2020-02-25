from typing import Any, Callable, List, Dict, Optional, Union

import torch
import torch.nn as nn

from .model_partitioning import METIS_partition
from .compiler import compile_partitoned_model
from .model_profiling import Graph, profile_network, build_graph, Profile, NodeWeightFunction, EdgeWeightFunction
from .pipeline import Pipeline, PipelineConfig, StageConfig, SyncBuffersMode
from .utils import Devices, Tensors

__all__ = [
    'pipe_model', 'profile_network', 'build_graph', 'partition_model',
    'METIS_partition', 'Pipeline'
]


def pipe_model(model: nn.Module,
               sample_batch: Tensors,
               kwargs: Optional[Dict] = None,
               n_iter=10,
               nparts: int = 4,
               depth=1000,
               basic_blocks: Optional[List[nn.Module]] = None,
               node_weight_function: Optional[NodeWeightFunction] = None,
               edge_weight_function: Optional[EdgeWeightFunction] = None,
               use_layers_only_graph: bool = False,
               output_file: str = None,
               DEBUG=False,
               recomputation=False,
               save_memory_mode=False,
               METIS_opt=dict()) -> Graph:
    '''attemps to partition a model to given number of parts using our profiler
       this will produce a python file with the partition config

    the generated python file exposes a method named create_pipeline_configuration which can be consumed by Pipeline or by the user directly
    for this specific model config

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    kwargs:
        aditional kwargs dictionary to pass to the model
    n_iter:
        number of profiling iteration used to gather statistics
    nparts:
        the number of partitions
    depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    node_weight_function:
            an optional weight function for the nodes should be a function from Node to int
            if not given a default weight of 1 will be given to all nodes
    edge_weight_function:
            an optional weight function for the edges should be a function (Node,Node) to int
            if not given a default value of 1 will be given to all edges
    use_layers_only_graph:
        whether to partition a smaller version of the graph containing only the layers (usefull fo big models with lots of unprofiled ops)
    output_file:
        the file name in which to save the partition config
        if not given defualts to generated_{modelClass}{actualNumberOfPartitions}
    DEBUG:
        whether to generate the debug version of the partition more comments and assertions in the generated file
    METIS_opt:
        dict of additional kwargs to pass to the METIS partitioning algorithm
    '''

    graph = partition_model(model,
                            sample_batch,
                            kwargs=kwargs,
                            max_depth=depth,
                            n_iter=n_iter,
                            nparts=nparts,
                            basic_blocks=basic_blocks,
                            node_weight_function=node_weight_function,
                            edge_weight_function=edge_weight_function,
                            use_layers_only_graph=use_layers_only_graph,
                            recomputation=recomputation,
                            save_memory_mode=save_memory_mode,
                            METIS_opt=METIS_opt)

    compile_partitoned_model(graph,
                             model,
                             output_file=output_file,
                             verbose=DEBUG)

    return graph


def partition_model(model: nn.Module,
                    sample_batch: Tensors,
                    kwargs: Optional[Dict] = None,
                    n_iter=10,
                    nparts=4,
                    max_depth=100,
                    basic_blocks: Optional[List[nn.Module]] = None,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_only_graph: bool = False,
                    recomputation: bool = False,
                    save_memory_mode: bool = False,
                    METIS_opt=dict()) -> Graph:
    '''
    profiles the network and return a graph representing the partition

    Parameters:
    model:
        the network we wish to model
    sample_batch:
        a sample input to use for tracing
    kwargs:
        aditional kwargs dictionary to pass to the model
    n_iter:
        number of profiling iteration used to gather statistics
    nparts:
        the number of partitions
    max_depth:
        how far down we go in the model tree determines the detail level of the graph
    basic_blocks:
        an optional list of modules that if encountered will not be broken down
    node_weight_function:
        an optional weight function for the nodes should be a function from Node to int
        if not given a default weight of 1 will be given to all nodes
    edge_weight_function:
        an optional weight function for the edges should be a function (Node,Node) to int
        if not given a default value of 1 will be given to all edges
    use_layers_only_graph:
        whether to partition a smaller version of the graph containing only the layers (usefull fo big models with lots of unprofiled ops)
    METIS_opt:
        dict of additional kwargs to pass to the METIS partitioning algorithm
    '''
    graph = build_graph(model,
                        sample_batch,
                        kwargs=kwargs,
                        max_depth=max_depth,
                        basic_blocks=basic_blocks,
                        n_iter=n_iter,
                        recomputation=recomputation,
                        save_memory_mode=save_memory_mode)

    graph = METIS_partition(graph,
                            nparts,
                            node_weight_function=node_weight_function,
                            edge_weight_function=edge_weight_function,
                            use_layers_only_graph=use_layers_only_graph,
                            **METIS_opt)

    return graph
