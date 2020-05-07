from typing import Any, Callable, List, Dict, Optional, Union

import torch
import torch.nn as nn

from .model_partitioning import METIS_partition
from .compiler import compile_partitioned_model
from .model_profiling import Graph, profile_network, LayerProfiler, trace_module, Profile, NodeWeightFunction, EdgeWeightFunction
from .model_profiling.graph_executor import execute_graph
from .pipeline import Pipeline, PipelineConfig, StageConfig, SyncBuffersMode
from .utils import Devices, Tensors

__all__ = [
    'pipe_model', 'profile_network', 'trace_module', 'partition_model',
    'METIS_partition', 'Pipeline'
]


def pipe_model(model: nn.Module,
               batch_dim: int,
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
               generate_model_parallel: bool = False,
               recomputation=False,
               METIS_opt=dict(),
               force_no_recomp_scopes=lambda s: False,
               save_memory_mode=False,
               use_layer_profiler=False,
               use_network_profiler=True,) -> Graph:
    '''attemps to partition a model to given number of parts using our profiler
       this will produce a python file with the partition config

    the generated python file exposes a method named create_pipeline_configuration which can be consumed by Pipeline or by the user directly
    for this specific model config

    Parameters:
    model:
        the network we wish to model
     batch_dim:
        the batch dimention of the sample batch
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
     generate_model_parallel:
        whether to generate a model parallel version of the partition in the addition to the partitions themselves
    METIS_opt:
        dict of additional kwargs to pass to the METIS partitioning algorithm
    force_no_recomp_scopes:
        fn(scope):
            returns true if we want to force recomputation scope_specific_recomp
        defaut is lambda x: False
    '''

    if basic_blocks is None:
        basic_blocks = ()

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
                            METIS_opt=METIS_opt,
                            force_no_recomp_scopes=force_no_recomp_scopes,
                            use_layer_profiler=use_layer_profiler,
                            use_network_profiler=use_network_profiler,
                            save_memory_mode=save_memory_mode
                            )

    compile_partitioned_model(graph,
                              model,
                              batch_dim,
                              output_file=output_file,
                              generate_model_parallel=generate_model_parallel)

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
                    METIS_opt=dict(),
                    force_no_recomp_scopes=lambda s: False,
                    use_layer_profiler=False,
                    use_network_profiler=True,
                    save_memory_mode=False) -> Graph:
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
    if basic_blocks is None:
        basic_blocks = ()
    # TODO profiling integration
    graph = build_graph(model,
                        args=sample_batch,
                        kwargs=kwargs,
                        max_depth=max_depth,
                        basic_blocks=basic_blocks,
                        n_iter=n_iter,
                        use_layer_profiler=use_layer_profiler,
                        use_network_profiler=use_network_profiler,
                        recomputation=recomputation,
                        force_no_recomp_scopes=force_no_recomp_scopes,
                        save_memory_mode=save_memory_mode)

    graph = METIS_partition(graph,
                            nparts,
                            node_weight_function=node_weight_function,
                            edge_weight_function=edge_weight_function,
                            use_layers_only_graph=use_layers_only_graph,
                            **METIS_opt)

    return graph


def build_graph(model, args=(), kwargs=None, use_network_profiler=True, use_layer_profiler=False, save_memory_mode=False, recomputation=False, n_iter=10, max_depth=1000, basic_blocks=None, force_no_recomp_scopes=None):
    if basic_blocks is None:
        basic_blocks = ()
    if kwargs is None:
        kwargs = dict()

    graph = trace_module(model, args=args, kwargs=kwargs, depth=max_depth,
                         basic_blocks=basic_blocks)
    weights = None

    if use_layer_profiler:
        assert not save_memory_mode, "save memory mode is not supported for LayerProfiler"
        profiler = LayerProfiler(recomputation=recomputation, n_iter=n_iter,
                                 force_no_recomp_scopes=force_no_recomp_scopes)
        execute_graph(model, graph, model_args=args, model_kwargs=kwargs,
                      pre_hook=profiler.time_forward, post_hook=profiler.time_backward)
        weights = profiler.get_weights()
    elif use_network_profiler:
        weights = profile_network(model, args, kwargs=kwargs,
                                  basic_blocks=basic_blocks,
                                  max_depth=max_depth,
                                  n_iter=n_iter,
                                  save_memory_mode=False,
                                  recomputation=recomputation,
                                  save_memory_mode=save_memory_mode,
                                  force_no_recomp_scopes=force_no_recomp_scopes)
    if not (weights is None):
        for n in graph.nodes:
            n.weight = weights.get(n.scope, n.weight)

    return graph
