import functools
import warnings
from typing import Optional, Dict, Union, List, Tuple, Callable

import torch
from torch import nn as nn

from autopipe.autopipe import NodeWeightFunction, EdgeWeightFunction, Graph, compile_partitioned_model, \
    compute_and_maybe_cache, GraphCache, partition_and_match_weights_until_last_partition_is_with_no_recomputation, \
    ExecTimes, FullExecTimes, metis_partition, acyclic_partition, partition_2dbin_pack, partition_mpipe, \
    compute_and_cache, move_tensors, trace_module, infer_req_grad, GraphProfiler, pre_hook_factory, post_hook_factory, \
    execute_graph, profile_network
from autopipe.autopipe.model_partitioning.pipedream.pipedream_partition_no_hir import partition_pipedream
from autopipe.autopipe.model_profiling import profiler


def pipe_model(model: nn.Module, batch_dim: int, model_args: tuple = (), model_kwargs: Optional[Dict] = None,
               n_iter: int = 10, nparts: int = 4, depth: int = 1000,
               basic_blocks: Optional[Union[List[nn.Module], Tuple[nn.Module]]] = None,
               node_weight_function: Optional[NodeWeightFunction] = None,
               edge_weight_function: Optional[EdgeWeightFunction] = None,
               use_layers_only_graph: bool = True,
               output_file: Optional[str] = None,
               generate_explicit_del: bool = False,
               generate_activation_propagation: bool = True,
               recomputation: bool = False,
               partitioning_method: str = "ACYCLIC",
               METIS_opt: Optional[Dict] = None,
               acyclic_opt: Optional[Dict] = None,
               binpack_opt: Optional[Dict] = None,
               mpipe_opt: Optional[Dict] = None,
               force_no_recomp_scopes: Optional[Callable[[str], bool]] = None,
               save_memory_mode: bool = False,
               trace_on_gpu=False,
               use_graph_profiler: bool = True,
               use_network_profiler: bool = False,
               profile_ops: bool = True,
               graph: Optional[Graph] = None,
               async_pipe=False,
               trace_cache_name=None,
               profiles_cache_name=None,
               dont_use_async_meta_alg=False) -> Graph:
    """
    Attempts to partition a model to given number of bins (parts).
    This will produce a python file with the partitions and config.

    Parameters:
    ------------
    model:
        the network we wish to model
     batch_dim:
        the batch dimension of the sample batch
    model_args:
        a sample input to use for tracing
    model_kwargs:
        additional kwargs dictionary to pass to the model
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
        if not given defaults to generated_{modelClass}{actualNumberOfPartitions}
    generate_explicit_del:
        whether to generate del statements to explicitly delete variables when they are no longer used
        default False
    generate_activation_propagation:
        in cases where a stage sends an activation to multiple stages.
        for example 0->[1,3,4]
        decide whether to have each stage send the activation to the next target
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
        default is lambda x: False
    use_graph_profiler:
        whether to use the new graph based profiler
        default True
    use_network_profiler:
        whether to use the older model based network_profiler
        default False
    profile_ops:
        whether to also profile ops when using the GraphProfiler
        default True
    save_memory_mode:
        minimize memory footprint during profiling
        sacrifice speed for memory
        default False
    graph:
        an existing graph to repartition
        default None
    """

    if basic_blocks is None:
        basic_blocks = ()

    graph = partition_model(model, model_args=model_args, model_kwargs=model_kwargs, n_iter=n_iter, nparts=nparts,
                            max_depth=depth, basic_blocks=basic_blocks, node_weight_function=node_weight_function,
                            edge_weight_function=edge_weight_function, use_layers_only_graph=use_layers_only_graph,
                            recomputation=recomputation, partitioning_method=partitioning_method, METIS_opt=METIS_opt,
                            acyclic_opt=acyclic_opt, binpack_opt=binpack_opt, mpipe_opt=mpipe_opt,
                            force_no_recomp_scopes=force_no_recomp_scopes, use_graph_profiler=use_graph_profiler,
                            use_network_profiler=use_network_profiler, profile_ops=profile_ops,
                            save_memory_mode=save_memory_mode, trace_on_gpu=trace_on_gpu, graph=graph,
                            async_pipe=async_pipe,
                            trace_cache_name=trace_cache_name, profiles_cache_name=profiles_cache_name,
                            dont_use_async_meta_alg=dont_use_async_meta_alg)

    compile_partitioned_model(graph,
                              model,
                              batch_dim,
                              output_file=output_file,
                              generate_explicit_del=generate_explicit_del,
                              generate_activation_propagation=generate_activation_propagation)
    print("-I- generated code")
    return graph


def partition_model(model: nn.Module, model_args: tuple = (), model_kwargs: Optional[Dict] = None, n_iter: int = 10,
                    nparts: int = 4, max_depth: int = 100,
                    basic_blocks: Optional[Union[List[nn.Module], Tuple[nn.Module]]] = None,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None, use_layers_only_graph: bool = True,
                    recomputation: bool = True, partitioning_method: str = "ACYCLIC",
                    METIS_opt: Optional[Dict] = None,
                    acyclic_opt: Optional[Dict] = None,
                    binpack_opt: Optional[Dict] = None,
                    mpipe_opt: Optional[Dict] = None,
                    force_no_recomp_scopes: Optional[Callable[[str], bool]] = None, use_graph_profiler: bool = True,
                    use_network_profiler: bool = False, profile_ops: bool = True, save_memory_mode: bool = False,
                    trace_on_gpu=False,
                    graph: Optional[Graph] = None, use_virtual_stages: bool = True,
                    async_pipe=False,
                    trace_cache_name=None,
                    profiles_cache_name=None,
                    dont_use_async_meta_alg=False
                    ) -> Graph:
    """
    profiles the network and return a graph representing the partition

    Parameters:
    -------------
    model:
        the network we wish to model
    model_args:
        a sample input to use for tracing
    model_kwargs:
        additional kwargs dictionary to pass to the model
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
        whether to also profile ops when using the GraphProfiler
        default True
    save_memory_mode:
        minimize memory footprint during profiling
        sacrifice speed for memory
        default False
    graph:
        an existing graph to repartition
        default None
    use_virtual_stages:
        will try to use virtual stages if partitioning method supports it.
    """

    if basic_blocks is None:
        basic_blocks = ()
    if METIS_opt is None:
        METIS_opt = dict()
    if acyclic_opt is None:
        acyclic_opt = dict()
    if binpack_opt is None:
        binpack_opt = dict()
    if mpipe_opt is None:
        mpipe_opt = dict()

    if not async_pipe or not recomputation or dont_use_async_meta_alg:
        if graph is None:
            graph = compute_and_maybe_cache(build_profiled_graph, profiles_cache_name,
                                            model, _cache_cls_to_use=GraphCache, model_args=model_args,
                                            model_kwargs=model_kwargs,
                                            use_network_profiler=use_network_profiler,
                                            use_graph_profiler=use_graph_profiler,
                                            save_memory_mode=save_memory_mode,
                                            trace_on_gpu=trace_on_gpu,
                                            profile_ops=profile_ops,
                                            recomputation=recomputation, n_iter=n_iter, max_depth=max_depth,
                                            basic_blocks=basic_blocks, force_no_recomp_scopes=force_no_recomp_scopes,
                                            trace_cache_name=trace_cache_name)

        if nparts > 1:
            warnings.warn("PROFILER IS NOT USING MEMORY USAGE FOR NODES")
            # profiler.set_max_memory_usage(graph)
            GraphProfiler.activations_estimated_set_max_memory_usage(graph)  # TODO: this is far from perfect

            graph = partition_profiled_graph(graph, model, nparts, partitioning_method, node_weight_function,
                                             edge_weight_function, use_virtual_stages, use_layers_only_graph, METIS_opt,
                                             acyclic_opt, binpack_opt, mpipe_opt)

    else:
        # This requires heterogeneous profiling.
        # NOTE: very similar thing can be done to partition for heterogeneous accelerators.

        graph = build_graph_with_nparams_and_grad_reqs(model, model_args, model_kwargs, max_depth,
                                                       basic_blocks, save_memory_mode, trace_on_gpu,
                                                       res_cache_name=trace_cache_name)

        # FIXME: the mem is not saved in cache now.
        weights = compute_and_maybe_cache(get_full_profiles, profiles_cache_name,
                                          graph, model, model_args, model_kwargs, n_iter, profile_ops, max_depth,
                                          basic_blocks, force_no_recomp_scopes, save_memory_mode, use_graph_profiler,
                                          use_network_profiler)

        warnings.warn("PROFILER IS NOT USING MEMORY USAGE FOR NODES")
        # profiler.set_max_memory_usage(graph)
        GraphProfiler.activations_estimated_set_max_memory_usage(graph)  # TODO: this is far from perfect

        partition_profiled_graph_fn = functools.partial(partition_profiled_graph, model=model, nparts=nparts,
                                                        partitioning_method=partitioning_method,
                                                        node_weight_function=node_weight_function,
                                                        edge_weight_function=edge_weight_function,
                                                        use_virtual_stages=use_virtual_stages,
                                                        use_layers_only_graph=use_layers_only_graph,
                                                        METIS_opt=METIS_opt,
                                                        acyclic_opt=acyclic_opt, binpack_opt=binpack_opt,
                                                        mpipe_opt=mpipe_opt)

        graph = partition_and_match_weights_until_last_partition_is_with_no_recomputation(graph, weights,
                                                                                          partitioning_method,
                                                                                          partition_profiled_graph_fn)

    return graph


def get_full_profiles(graph, model, model_args, model_kwargs, n_iter, profile_ops, max_depth, basic_blocks,
                      force_no_recomp_scopes, save_memory_mode, use_graph_profiler, use_network_profiler):
    print("-I- profiling model (recomp)")
    recomputation_times = get_profiles(graph,
                                       model,
                                       model_args=model_args,
                                       model_kwargs=model_kwargs,
                                       use_network_profiler=use_network_profiler,
                                       use_graph_profiler=use_graph_profiler,
                                       save_memory_mode=save_memory_mode,
                                       profile_ops=profile_ops, recomputation=True,
                                       n_iter=n_iter,
                                       max_depth=max_depth,
                                       basic_blocks=basic_blocks,
                                       force_no_recomp_scopes=force_no_recomp_scopes)
    print("-I- profiling model (no recomp)")
    no_recomputation_times = get_profiles(graph,
                                          model,
                                          model_args=model_args,
                                          model_kwargs=model_kwargs,
                                          use_network_profiler=use_network_profiler,
                                          use_graph_profiler=use_graph_profiler,
                                          save_memory_mode=save_memory_mode,
                                          profile_ops=profile_ops, recomputation=False,
                                          n_iter=n_iter,
                                          max_depth=max_depth,
                                          basic_blocks=basic_blocks,
                                          force_no_recomp_scopes=force_no_recomp_scopes)
    for n in graph.nodes:
        if n.scope not in no_recomputation_times:
            no_recomputation_times[n.scope] = ExecTimes(0, 0)
        if n.scope not in recomputation_times:
            recomputation_times[n.scope] = ExecTimes(0, 0)
    weights = {
        n.id: FullExecTimes(recomputation_times[n.scope],
                            no_recomputation_times[n.scope])
        for n in graph.nodes
    }
    print("-I- model profiled")
    return weights


def partition_profiled_graph(graph, model, nparts, partitioning_method, node_weight_function, edge_weight_function,
                             use_virtual_stages, use_layers_only_graph, METIS_opt, acyclic_opt, binpack_opt, mpipe_opt):
    partitioning_method = partitioning_method.lower()
    if partitioning_method == "metis":
        print("-I- using METIS partitioning algorithm")
        graph = metis_partition(graph,
                                nparts,
                                node_weight_function=node_weight_function,
                                edge_weight_function=edge_weight_function,
                                use_layers_only_graph=use_layers_only_graph,
                                use_virtual_stages=use_virtual_stages,
                                **METIS_opt)
    elif partitioning_method == "acyclic":
        print("-I- using Acyclic Partitioning algorithm")
        acyclic_partition(model, graph, nparts,
                          node_weight_function=node_weight_function,
                          edge_weight_function=edge_weight_function,
                          use_layers_graph=use_layers_only_graph,
                          **acyclic_opt)
    elif partitioning_method == "2dbin":
        if "n_clusters" not in binpack_opt:
            if "analyze_n_clusters" not in binpack_opt:
                warnings.warn(
                    "expected --n_clusters or --analyze_n_clusters to be given to binpack_opt. will set n_clusters=2 as default")
            binpack_opt["n_clusters"] = 2
        else:
            warnings.warn("Will infer `n_clusters` with user assistance")

        graph, stage_to_gpu_map = partition_2dbin_pack(graph, num_gpus=nparts,
                                                       node_weight_function=node_weight_function,
                                                       **binpack_opt)
    elif partitioning_method == "mpipe":
        graph, stage_to_gpu_map = partition_mpipe(model, graph, num_gpus=nparts,
                                                  node_weight_function=node_weight_function,
                                                  edge_weight_function=edge_weight_function,
                                                  **mpipe_opt)

    elif partitioning_method == "pipedream":
        t1 = node_weight_function.MULT_FACTOR
        t2 = edge_weight_function.MULT_FACTOR
        assert t1 == t2
        warnings.warn("forcing mult factor to 1")
        node_weight_function.MULT_FACTOR = 1.0
        edge_weight_function.MULT_FACTOR = 1.0
        t3 = edge_weight_function.ensure_positive
        edge_weight_function.ensure_positive = False
        graph = partition_pipedream(graph, num_gpus=nparts,
                                    node_weight_function=node_weight_function,
                                    edge_weight_function=edge_weight_function,
                                    num_machines_in_first_level=None,
                                    )

        node_weight_function.MULT_FACTOR = t1
        edge_weight_function.MULT_FACTOR = t2
        edge_weight_function.ensure_positive = t3
        del t1
        del t2
        del t3


    else:
        raise NotImplementedError(partitioning_method)  # shouldn't happen
    return graph


def build_profiled_graph(model: nn.Module,
                         model_args: tuple = (),
                         model_kwargs: Optional[Dict] = None,
                         use_network_profiler: bool = False,
                         use_graph_profiler: bool = True,
                         save_memory_mode: bool = False,
                         trace_on_gpu=False,
                         profile_ops: bool = True,
                         recomputation: bool = False,
                         n_iter: int = 10,
                         max_depth: int = 1000,
                         basic_blocks: Optional[List[nn.Module]] = None,
                         force_no_recomp_scopes: Optional[Callable[[str], bool]] = None,
                         trace_cache_name=None) -> Graph:
    """
     Builds a graph representation of the model.
     Profiles execution times of model's operations (nodes)
     Infers gradient requirements for nodes.

    The representation is semantically identical to the forward pass.

    Parameters:
    ------------------
    model:
        the network we wish to model
    args:
        a sample input to use for tracing
    kwargs:
        additional kwargs dictionary to pass to the model
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
        whether to partition a smaller version of the graph containing only the layers (useful fo big models with lots of unprofiled ops)
    use_graph_profiler:
        whether to use the new graph based profiler
        default True
    use_network_profiler:
        whether to use the older model based network_profiler
        default False
    profile_ops:
        whether to also profile ops when using the GraphProfiler
        default True
    save_memory_mode:
        minimize memory footprint during profiling
        sacrifice speed for memory
        default False
    """

    graph = build_graph_with_nparams_and_grad_reqs(model, model_args, model_kwargs, max_depth,
                                                   basic_blocks, save_memory_mode, trace_on_gpu, res_cache_name=trace_cache_name)


    print("-I- profiling model")
    weights = get_profiles(graph,
                           model,
                           model_args=model_args,
                           model_kwargs=model_kwargs,
                           use_network_profiler=use_network_profiler,
                           use_graph_profiler=use_graph_profiler,
                           save_memory_mode=save_memory_mode,
                           profile_ops=profile_ops, recomputation=recomputation,
                           n_iter=n_iter,
                           max_depth=max_depth,
                           basic_blocks=basic_blocks,
                           force_no_recomp_scopes=force_no_recomp_scopes)
    for n in graph.nodes:
        n.weight = weights.get(n.scope, ExecTimes(0, 0))

    print("-I- model profiled")

    return graph


def build_graph_with_nparams_and_grad_reqs(model, model_args, model_kwargs, max_depth, basic_blocks, save_memory_mode, trace_on_gpu,
                                           res_cache_name=None) -> Graph:
    if res_cache_name:
        return compute_and_cache(build_graph_with_nparams_and_grad_reqs, res_cache_name, model, model_args, model_kwargs, max_depth,
                                 basic_blocks, save_memory_mode, trace_on_gpu, res_cache_name=None,
                                 _cache_cls_to_use=GraphCache)

    # dev WARNING: can move , model, model_args, model_kwargs to CPU
    if save_memory_mode:
        if not trace_on_gpu:
            print("-I- tracing on CPU")
            with torch.no_grad():
                model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cpu')
        else:
            print("-I- tracing on GPU")
            with torch.no_grad():
                model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cuda')

    model_device = next(model.parameters()).device
    print(f"-I- tracing device: {model_device}")

    print("-I- tracing model")
    graph = trace_module(model, args=model_args, kwargs=model_kwargs, depth=max_depth,
                         basic_blocks=basic_blocks)
    print("-I- graph built")
    # TODO: tracing is sometimes done on cpu...
    print("-I- inferring gradient requirements")
    if save_memory_mode:
        with torch.no_grad():
            model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cpu')
    infer_req_grad(graph, model, args=model_args, kwargs=model_kwargs)
    print("-I- inferred gradient requirements")

    print("-I- inferring params per node")
    graph.calculate_params_per_node(model)

    return graph


def get_profiles(graph: Graph, model: nn.Module,
                 model_args: tuple = (),
                 model_kwargs: Optional[Dict] = None,
                 use_network_profiler: bool = False, use_graph_profiler: bool = True,
                 save_memory_mode: bool = False, profile_ops: bool = True, recomputation: bool = False,
                 n_iter: int = 10, max_depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None,
                 force_no_recomp_scopes: Optional[Callable[[str], bool]] = None
                 ):
    if basic_blocks is None:
        basic_blocks = ()
    if model_kwargs is None:
        model_kwargs = dict()

    if use_graph_profiler:
        print(
            f"-I- using graph profiler with op profiling = {profile_ops} save_memory_mode = {save_memory_mode}")

        if save_memory_mode:
            model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cpu')

        # torch.cuda.reset_max_memory_allocated()
        profiler = GraphProfiler(recomputation=recomputation, n_iter=n_iter, profile_ops=profile_ops,
                                 force_no_recomp_scopes=force_no_recomp_scopes, save_memory_mode=save_memory_mode)
        pre_hook = pre_hook_factory(profiler.time_forward)
        post_hook = post_hook_factory(profiler.time_backward)
        execute_graph(model, graph, model_args=model_args, model_kwargs=model_kwargs,
                      pre_hook=pre_hook, post_hook=post_hook, enforce_out_of_place=True)

        # warnings.warn("PROFILER IS NOT USING MEMORY USAGE FOR NODES")
        # # profiler.set_max_memory_usage(graph)
        # profiler.activations_estimated_set_max_memory_usage(graph)  # TODO: this is far from perfect

        # print(f"-I- profiling mem {torch.cuda.max_memory_allocated() / 1e9} GB")

        weights = profiler.get_weights()


    elif use_network_profiler:
        warnings.warn("network profiler is deprecated, use graph profiler")
        print(
            f"-I- using network profiler with save_memory_mode = {save_memory_mode}")
        assert not profile_ops, "op profiling is not supported in the network profiler"
        # TODO: deprecated
        weights = profile_network(model, model_args, kwargs=model_kwargs,
                                  basic_blocks=basic_blocks,
                                  max_depth=max_depth,
                                  n_iter=n_iter,
                                  recomputation=recomputation,
                                  save_memory_mode=save_memory_mode,
                                  force_no_recomp_scopes=force_no_recomp_scopes)
    else:
        raise ValueError("missing profiling method")

    assert weights is not None

    return weights