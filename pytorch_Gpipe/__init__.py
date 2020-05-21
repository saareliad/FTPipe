from typing import Any, Callable, List, Dict, Optional, Union

import torch
import torch.nn as nn

from .model_partitioning import METIS_partition
from .compiler import compile_partitioned_model
from .model_profiling import Graph, profile_network, GraphProfiler, trace_module, ExecTimes, NodeWeightFunction, EdgeWeightFunction
from .model_profiling.graph_executor import execute_graph
from .pipeline import Pipeline, PipelineConfig, StageConfig, SyncBuffersMode
from .utils import move_tensors

__all__ = [
    'pipe_model', 'profile_network', 'trace_module', 'partition_model',
    'METIS_partition', 'Pipeline'
]

# TODO document everything. too many things have changed since last documantation pass


def pipe_model(model: nn.Module,
               batch_dim: int,
               args: tuple = (),
               kwargs: Optional[Dict] = None,
               n_iter=10,
               nparts: int = 4,
               depth=1000,
               basic_blocks: Optional[List[nn.Module]] = None,
               node_weight_function: Optional[NodeWeightFunction] = None,
               edge_weight_function: Optional[EdgeWeightFunction] = None,
               use_layers_only_graph: bool = True,
               output_file: str = None,
               generate_model_parallel: bool = False,
               generate_explicit_del=False,
               recomputation=False,
               METIS_opt=dict(),
               force_no_recomp_scopes=lambda s: False,
               save_memory_mode=False,
               use_graph_profiler=False,
               use_network_profiler=True,
               profile_ops=False) -> Graph:
    '''attemps to partition a model to given number of parts using our profiler
       this will produce a python file with the partition config

    the generated python file exposes a method named create_pipeline_configuration which can be consumed by Pipeline or by the user directly
    for this specific model config

    Parameters:
    ------------
    model:
        the network we wish to model
     batch_dim:
        the batch dimention of the sample batch
    args:
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
    generate_explicit_del:
        whether to generate del statements to explicitly delete variables when they are no longer used
        default False
    METIS_opt:
        dict of additional kwargs to pass to the METIS partitioning algorithm
    force_no_recomp_scopes:
        fn(scope):
            returns true if we want to force recomputation scope_specific_recomp
        defaut is lambda x: False
    use_graph_profiler:
        whether to use the new graph based profiler
        default False
    use_network_profiler:
        whether to use the older model based network_profiler
        default True
    profile_ops:
        weheter to also profile ops when using the GraphProfiler
        default False
    save_memory_mode:
        minimize memory footprint during profiling
        sacrifice speed for memory
        default False
    '''

    if basic_blocks is None:
        basic_blocks = ()

    graph = partition_model(model,
                            args=args,
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
                            use_graph_profiler=use_graph_profiler,
                            use_network_profiler=use_network_profiler,
                            profile_ops=profile_ops,
                            save_memory_mode=save_memory_mode
                            )

    compile_partitioned_model(graph,
                              model,
                              batch_dim,
                              output_file=output_file,
                              generate_explicit_del=generate_explicit_del,
                              generate_model_parallel=generate_model_parallel)
    print("generated code")
    return graph


def partition_model(model: nn.Module,
                    args: tuple = (),
                    kwargs: Optional[Dict] = None,
                    n_iter=10,
                    nparts=4,
                    max_depth=100,
                    basic_blocks: Optional[List[nn.Module]] = None,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_only_graph: bool = True,
                    recomputation: bool = False,
                    METIS_opt=dict(),
                    force_no_recomp_scopes=lambda s: False,
                    use_graph_profiler=False,
                    use_network_profiler=True,
                    profile_ops=False,
                    save_memory_mode=False) -> Graph:
    '''
    profiles the network and return a graph representing the partition

    Parameters:
    -------------
    model:
        the network we wish to model
    args:
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
    use_graph_profiler:
        whether to use the new graph based profiler
        default False
    use_network_profiler:
        whether to use the older model based network_profiler
        default True
    profile_ops:
        weheter to also profile ops when using the GraphProfiler
        default False
    '''
    if basic_blocks is None:
        basic_blocks = ()
    graph = build_graph(model,
                        args=args,
                        kwargs=kwargs,
                        max_depth=max_depth,
                        basic_blocks=basic_blocks,
                        n_iter=n_iter,
                        use_graph_profiler=use_graph_profiler,
                        use_network_profiler=use_network_profiler,
                        profile_ops=profile_ops,
                        recomputation=recomputation,
                        force_no_recomp_scopes=force_no_recomp_scopes,
                        save_memory_mode=save_memory_mode)

    graph = METIS_partition(graph,
                            nparts,
                            node_weight_function=node_weight_function,
                            edge_weight_function=edge_weight_function,
                            use_layers_only_graph=use_layers_only_graph,
                            **METIS_opt)
    print("partitioned model")

    return graph


def build_graph(model: nn.Module, args: tuple = (), kwargs: Optional[Dict] = None,
                use_network_profiler: bool = True, use_graph_profiler: bool = False,
                save_memory_mode: bool = False, profile_ops: bool = False,
                recomputation: bool = False, n_iter: int = 10,
                max_depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None,
                force_no_recomp_scopes: Optional[Callable[[str], bool]] = None) -> Graph:
    """
    builds a graph representation of the model which is semantically identical to the forward pass
    optionaly can also profiler execution times of the model's oprations

    Parameters:
    ------------------
    model:
        the network we wish to model
    args:
        a sample input to use for tracing
    kwargs:
        aditional kwargs dictionary to pass to the model
    n_iter:
        number of profiling iteration used to gather statistics
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
    use_graph_profiler:
        whether to use the new graph based profiler
        default False
    use_network_profiler:
        whether to use the older model based network_profiler
        default True
    profile_ops:
        weheter to also profile ops when using the GraphProfiler
        default False
    save_memory_mode:
        minimize memory footprint during profiling
        sacrifice speed for memory
        default False
    """

    if basic_blocks is None:
        basic_blocks = ()
    if kwargs is None:
        kwargs = dict()

    graph = trace_module(model, args=args, kwargs=kwargs, depth=max_depth,
                         basic_blocks=basic_blocks)
    weights = None
    print("graph built")
    if use_graph_profiler:
        print(
            f"using graph profiler with op profiling = {profile_ops} save_memory_mode = {save_memory_mode}")

        if save_memory_mode:
            model, args, kwargs = move_tensors((model, args, kwargs), 'cpu')

        torch.cuda.reset_max_memory_allocated()
        profiler = GraphProfiler(recomputation=recomputation, n_iter=n_iter, profile_ops=profile_ops,
                                 force_no_recomp_scopes=force_no_recomp_scopes)
        execute_graph(model, graph, model_args=args, model_kwargs=kwargs,
                      pre_hook=profiler.time_forward, post_hook=profiler.time_backward)
        print(f"profiling mem {torch.cuda.max_memory_allocated()/1e9} GB")
        weights = profiler.get_weights()
    elif use_network_profiler:
        print(
            f"using network profiler with save_memory_mode = {save_memory_mode}")
        assert not profile_ops, "op profiling is not supported in the network profiler"
        weights = profile_network(model, args, kwargs=kwargs,
                                  basic_blocks=basic_blocks,
                                  max_depth=max_depth,
                                  n_iter=n_iter,
                                  recomputation=recomputation,
                                  save_memory_mode=save_memory_mode,
                                  force_no_recomp_scopes=force_no_recomp_scopes)
    if not (weights is None):
        print("model profiled")
        for n in graph.nodes:
            n.weight = weights.get(n.scope, ExecTimes(0, 0))

    return graph
