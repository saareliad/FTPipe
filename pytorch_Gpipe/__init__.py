from typing import Callable, List, Dict, Optional

import torch
import torch.nn as nn

from .compiler import compile_partitioned_model
from .model_partitioning import METIS_partition, acyclic_partition, partition_2dbin_pack, determine_n_clusters, \
    get_weight_functions
from .model_profiling import Graph, profile_network, GraphProfiler, trace_module, ExecTimes, NodeWeightFunction, \
    EdgeWeightFunction
from .model_profiling.graph_executor import execute_graph
from .model_profiling.infer_req_grad import infer_req_grad
from .utils import move_tensors

__all__ = [
    'pipe_model', 'profile_network', 'trace_module', 'partition_model',
    'METIS_partition'
]


# TODO document everything. too many things have changed since last documantation pass


def pipe_model(model: nn.Module,
               batch_dim: int,
               args: tuple = (),
               kwargs: Optional[Dict] = None,
               n_iter: int = 10,
               nparts: int = 4,
               depth: int = 1000,
               basic_blocks: Optional[List[nn.Module]] = None,
               node_weight_function: Optional[NodeWeightFunction] = None,
               edge_weight_function: Optional[EdgeWeightFunction] = None,
               use_layers_only_graph: bool = True,
               output_file: Optional[str] = None,
               generate_model_parallel: bool = False,
               generate_explicit_del: bool = False,
               generate_activation_propagation: bool = True,
               recomputation: bool = False,
               partitioning_method: str = "ACYCLIC",
               METIS_opt: Optional[Dict] = None,
               acyclic_opt: Optional[Dict] = None,
               binpack_opt: Optional[Dict] = None,
               force_no_recomp_scopes: Optional[Callable[[str], bool]] = None,
               save_memory_mode: bool = False,
               use_graph_profiler: bool = True,
               use_network_profiler: bool = False,
               profile_ops: bool = True,
               graph: Optional[Graph] = None) -> Graph:
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
    generate_activation_propagation:
        in cases where a stage sends an activation to multiple stages.
        for example 0->[1,3,4]
        decide wether to have each stage send the activation to the next target
        0->1->3->4
        or have it sent directly from the source
    partitioning_method:
        partitioning method to use
    METIS_opt:
        dict of additional kwargs to pass to the METIS partitioning algorithm
    acyclic_opt:
        dict of additional kwargs to pass to the acyclic partitioning algorithm
    binpack_opt:
        dict of additional kwargs to pass to the binback partitioning algorithm
    force_no_recomp_scopes:
        fn(scope):
            returns true if we want to force recomputation scope_specific_recomp
        defaut is lambda x: False
    use_graph_profiler:
        whether to use the new graph based profiler
        default True
    use_network_profiler:
        whether to use the older model based network_profiler
        default False
    profile_ops:
        weheter to also profile ops when using the GraphProfiler
        default True
    save_memory_mode:
        minimize memory footprint during profiling
        sacrifice speed for memory
        default False
    graph:
        an existing graph to repartition
        default None
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
                            partitioning_method=partitioning_method,
                            METIS_opt=METIS_opt,
                            acyclic_opt=acyclic_opt,
                            binpack_opt=binpack_opt,
                            force_no_recomp_scopes=force_no_recomp_scopes,
                            use_graph_profiler=use_graph_profiler,
                            use_network_profiler=use_network_profiler,
                            profile_ops=profile_ops,
                            save_memory_mode=save_memory_mode,
                            graph=graph
                            )

    compile_partitioned_model(graph,
                              model,
                              batch_dim,
                              output_file=output_file,
                              generate_explicit_del=generate_explicit_del,
                              generate_model_parallel=generate_model_parallel,
                              generate_activation_propagation=generate_activation_propagation)
    print("-I- generated code")
    return graph


def partition_model(model: nn.Module,
                    args: tuple = (),
                    kwargs: Optional[Dict] = None,
                    n_iter: int = 10,
                    nparts: int = 4,
                    max_depth: int = 100,
                    basic_blocks: Optional[List[nn.Module]] = None,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None,
                    use_layers_only_graph: bool = True,
                    recomputation: bool = True,
                    partitioning_method: str = "ACYCLIC",
                    METIS_opt: Optional[Dict] = None,
                    acyclic_opt: Optional[Dict] = None,
                    binpack_opt:  Optional[Dict] = None,
                    force_no_recomp_scopes: Optional[Callable[[str], bool]] = None,
                    use_graph_profiler: bool = True,
                    use_network_profiler: bool = False,
                    profile_ops: bool = True,
                    save_memory_mode: bool = False,
                    graph: Optional[Graph] = None) -> Graph:
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
    partitioning_method:
        partitioning method to use
        default ACYCLIC
    METIS_opt:
        dict of additional kwargs to pass to the METIS partitioning algorithm
    acyclic_opt:
        dict of additional kwargs to pass to the acyclic partitioning algorithm
    binpack_opt:
        dict of additional kwargs to pass to the binpack partitioning algorithm
    use_graph_profiler:
        whether to use the new graph based profiler
        default True
    use_network_profiler:
        whether to use the older model based network_profiler
        default False
    profile_ops:
        weheter to also profile ops when using the GraphProfiler
        default True
    '''
    if basic_blocks is None:
        basic_blocks = ()
    if METIS_opt is None:
        METIS_opt = dict()
    if acyclic_opt is None:
        acyclic_opt = dict()
    if binpack_opt is None:
        binpack_opt = dict()

    if graph is None:
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
    if partitioning_method == "MEITS":
        print("-I- using METIS partitioning algorithm")
        graph = METIS_partition(graph,
                                nparts,
                                node_weight_function=node_weight_function,
                                edge_weight_function=edge_weight_function,
                                use_layers_only_graph=use_layers_only_graph,
                                **METIS_opt)
    elif partitioning_method == "ACYCLIC":
        print("-I- using Acyclic Partitioning algorithm")
        acyclic_partition(model, graph, nparts,
                          node_weight_function=node_weight_function,
                          edge_weight_function=edge_weight_function,
                          use_layers_graph=use_layers_only_graph,
                          **acyclic_opt)
    elif partitioning_method == "2DBIN":
        if "n_clusters" not in binpack_opt:
            binpack_opt["n_clusters"] = 2
        # TODO: determine nclusters
        graph, stage_to_gpu_map = partition_2dbin_pack(graph, num_gpus=nparts,
                                                       node_weight_function=node_weight_function, **binpack_opt)
    else:
        raise NotImplementedError()

    print("-I- partitioned model")

    return graph


def build_graph(model: nn.Module,
                args: tuple = (),
                kwargs: Optional[Dict] = None,
                use_network_profiler: bool = False,
                use_graph_profiler: bool = True,
                save_memory_mode: bool = False,
                profile_ops: bool = True,
                recomputation: bool = False,
                n_iter: int = 10,
                max_depth: int = 1000,
                basic_blocks: Optional[List[nn.Module]] = None,
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
        default True
    use_network_profiler:
        whether to use the older model based network_profiler
        default False
    profile_ops:
        weheter to also profile ops when using the GraphProfiler
        default True
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
    print("-I- graph built")
    if use_graph_profiler:
        print(
            f"-I- using graph profiler with op profiling = {profile_ops} save_memory_mode = {save_memory_mode}")

        if save_memory_mode:
            model, args, kwargs = move_tensors((model, args, kwargs), 'cpu')

        torch.cuda.reset_max_memory_allocated()
        profiler = GraphProfiler(recomputation=recomputation, n_iter=n_iter, profile_ops=profile_ops,
                                 force_no_recomp_scopes=force_no_recomp_scopes, save_memory_mode=save_memory_mode)
        execute_graph(model, graph, model_args=args, model_kwargs=kwargs,
                      pre_hook=profiler.time_forward, post_hook=profiler.time_backward, enforce_out_of_place=True)
        print(f"-I- profiling mem {torch.cuda.max_memory_allocated() / 1e9} GB")
        weights = profiler.get_weights()
    elif use_network_profiler:
        print(
            f"-I- using network profiler with save_memory_mode = {save_memory_mode}")
        assert not profile_ops, "op profiling is not supported in the network profiler"
        weights = profile_network(model, args, kwargs=kwargs,
                                  basic_blocks=basic_blocks,
                                  max_depth=max_depth,
                                  n_iter=n_iter,
                                  recomputation=recomputation,
                                  save_memory_mode=save_memory_mode,
                                  force_no_recomp_scopes=force_no_recomp_scopes)
    if not (weights is None):
        print("-I- model profiled")
        for n in graph.nodes:
            n.weight = weights.get(n.scope, ExecTimes(0, 0))

    infer_req_grad(graph, model, args=args, kwargs=kwargs)
    return graph
