import functools
import warnings
from collections import namedtuple
from typing import Callable, List, Dict, Optional, Union, Tuple
import numpy as np

import torch
import torch.nn as nn

from .cache_utils import compute_and_cache, compute_and_maybe_cache, PickleCache, GraphCache
from .compiler import compile_partitioned_model
from .model_partitioning import METIS_partition, acyclic_partition, partition_2dbin_pack, analyze_n_clusters, \
    get_weight_functions
from .model_profiling import Graph, Node, profile_network, GraphProfiler, trace_module, NodeWeightFunction, \
    EdgeWeightFunction
from .model_profiling.graph_executor import execute_graph, pre_hook_factory, post_hook_factory
from .model_profiling.infer_req_grad import infer_req_grad
from .utils import move_tensors, ExecTimes

FullExecTimes = namedtuple('FullExecTimes', 'recomputation no_recomputation')


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
               force_no_recomp_scopes: Optional[Callable[[str], bool]] = None,
               save_memory_mode: bool = False,
               trace_on_gpu=False,
               use_graph_profiler: bool = True,
               use_network_profiler: bool = False,
               profile_ops: bool = True,
               graph: Optional[Graph] = None,
               async_pipe=False,
               trace_cache_name=None,
               profiles_cache_name=None) -> Graph:
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
                            acyclic_opt=acyclic_opt, binpack_opt=binpack_opt,
                            force_no_recomp_scopes=force_no_recomp_scopes, use_graph_profiler=use_graph_profiler,
                            use_network_profiler=use_network_profiler, profile_ops=profile_ops,
                            save_memory_mode=save_memory_mode, trace_on_gpu=trace_on_gpu, graph=graph,
                            async_pipe=async_pipe,
                            trace_cache_name=trace_cache_name, profiles_cache_name=profiles_cache_name)

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
                    recomputation: bool = True, partitioning_method: str = "ACYCLIC", METIS_opt: Optional[Dict] = None,
                    acyclic_opt: Optional[Dict] = None, binpack_opt: Optional[Dict] = None,
                    force_no_recomp_scopes: Optional[Callable[[str], bool]] = None, use_graph_profiler: bool = True,
                    use_network_profiler: bool = False, profile_ops: bool = True, save_memory_mode: bool = False,
                    trace_on_gpu=False,
                    graph: Optional[Graph] = None, use_virtual_stages: bool = True,
                    async_pipe=False,
                    trace_cache_name=None,
                    profiles_cache_name=None,
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

    if not async_pipe or not recomputation:
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
            graph = partition_profiled_graph(graph, model, nparts, partitioning_method, node_weight_function,
                                             edge_weight_function, use_virtual_stages, use_layers_only_graph, METIS_opt,
                                             acyclic_opt, binpack_opt)

    else:
        # This requires heterogeneous profiling.
        # NOTE: very similar thing can be done to partition for heterogeneous accelerators.

        graph = build_graph_with_grad_reqs(model, model_args, model_kwargs, max_depth,
                                           basic_blocks, save_memory_mode, trace_on_gpu,
                                           res_cache_name=trace_cache_name)

        weights = compute_and_maybe_cache(get_full_profiles, profiles_cache_name,
                                          graph, model, model_args, model_kwargs, n_iter, profile_ops, max_depth,
                                          basic_blocks, force_no_recomp_scopes, save_memory_mode, use_graph_profiler,
                                          use_network_profiler)

        partition_profiled_graph_fn = functools.partial(partition_profiled_graph, model=model, nparts=nparts,
                                                        partitioning_method=partitioning_method,
                                                        node_weight_function=node_weight_function,
                                                        edge_weight_function=edge_weight_function,
                                                        use_virtual_stages=use_virtual_stages,
                                                        use_layers_only_graph=use_layers_only_graph,
                                                        METIS_opt=METIS_opt,
                                                        acyclic_opt=acyclic_opt, binpack_opt=binpack_opt)

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
                             use_virtual_stages, use_layers_only_graph, METIS_opt, acyclic_opt, binpack_opt):
    if partitioning_method == "METIS":
        print("-I- using METIS partitioning algorithm")
        graph = METIS_partition(graph,
                                nparts,
                                node_weight_function=node_weight_function,
                                edge_weight_function=edge_weight_function,
                                use_layers_only_graph=use_layers_only_graph,
                                use_virtual_stages=use_virtual_stages,
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
            if "analyze_n_clusters" not in binpack_opt:
                warnings.warn(
                    "expected --n_clusters or --analyze_n_clusters to be given to binpack_opt. will set n_clusters=2 as default")
            binpack_opt["n_clusters"] = 2
        else:
            warnings.warn("Will infer `n_clusters` with user assistance")

        graph, stage_to_gpu_map = partition_2dbin_pack(graph, num_gpus=nparts,
                                                       node_weight_function=node_weight_function,
                                                       **binpack_opt)
    else:
        raise NotImplementedError(partitioning_method)  # shouldn't happen
    return graph


def build_profiled_graph(model: nn.Module,
                         model_args: tuple = (),
                         model_kwargs: Optional[Dict] = None,
                         use_network_profiler: bool = False, use_graph_profiler: bool = True,
                         save_memory_mode: bool = False,
                         trace_on_gpu=False,
                         profile_ops: bool = True, recomputation: bool = False,
                         n_iter: int = 10, max_depth: int = 1000, basic_blocks: Optional[List[nn.Module]] = None,
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

    graph = build_graph_with_grad_reqs(model, model_args, model_kwargs, max_depth,
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
    print("-I- model profiled")
    for n in graph.nodes:
        n.weight = weights.get(n.scope, ExecTimes(0, 0))

    return graph


def build_graph_with_grad_reqs(model, model_args, model_kwargs, max_depth, basic_blocks, save_memory_mode, trace_on_gpu,
                               res_cache_name=None) -> Graph:
    if res_cache_name:
        return compute_and_cache(build_graph_with_grad_reqs, res_cache_name, model, model_args, model_kwargs, max_depth,
                                 basic_blocks, save_memory_mode, trace_on_gpu, res_cache_name=None,
                                 _cache_cls_to_use=GraphCache)

    # dev WARNING: can move , model, model_args, model_kwargs to CPU
    if save_memory_mode:
        if not trace_on_gpu:
            model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cpu')
        else:
            print("-I- tracing on GPU")
            model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cuda')

    print("-I- tracing model")
    graph = trace_module(model, args=model_args, kwargs=model_kwargs, depth=max_depth,
                         basic_blocks=basic_blocks)
    print("-I- graph built")
    # TODO: tracing is sometimes done on cpu...
    print("-I- inferring gradient requirements")
    if save_memory_mode:
        model, model_args, model_kwargs = move_tensors((model, model_args, model_kwargs), 'cpu')
    infer_req_grad(graph, model, args=model_args, kwargs=model_kwargs)
    print("-I- inferred gradient requirements")
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

        torch.cuda.reset_max_memory_allocated()
        profiler = GraphProfiler(recomputation=recomputation, n_iter=n_iter, profile_ops=profile_ops,
                                 force_no_recomp_scopes=force_no_recomp_scopes, save_memory_mode=save_memory_mode)
        pre_hook = pre_hook_factory(profiler.time_forward)
        post_hook = post_hook_factory(profiler.time_backward)
        execute_graph(model, graph, model_args=model_args, model_kwargs=model_kwargs,
                      pre_hook=pre_hook, post_hook=post_hook, enforce_out_of_place=True)
        print(f"-I- profiling mem {torch.cuda.max_memory_allocated() / 1e9} GB")
        weights = profiler.get_weights()
    elif use_network_profiler:
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


def partition_and_match_weights_until_last_partition_is_with_no_recomputation(graph: Graph,
                                                                              weights: Dict[Node, FullExecTimes],
                                                                              partitioning_method,
                                                                              partition_profiled_graph_fn,
                                                                              n_runs_limit=100):
    print("-I- partition_and_match_weights_until_last_partition_is_with_no_recomputation")
    allowed_mistakes = 0
    # HACK: allow mistakes for multilevel and acyclic...
    if partitioning_method == "ACYCLIC":
        allowed_mistakes += 2

    last_partition_scopes = set()
    current_mistakes = allowed_mistakes + 1
    n_runs = 0

    history = dict()
    while current_mistakes > allowed_mistakes and (n_runs_limit < 0 or n_runs < n_runs_limit):

        for n in graph.nodes:
            if n.scope in last_partition_scopes:
                n.weight = weights[n.id].no_recomputation
            else:
                n.weight = weights[n.id].recomputation

        n_runs += 1

        # Partition
        graph = partition_profiled_graph_fn(graph)

        # Load last partition last stage scopes
        last_p = max((n.stage_id for n in graph.nodes))
        generated_last_stage_scopes = [
            n.scope for n in graph.nodes if n.stage_id == last_p
        ]

        # Count mistakes (false positives and false negatives)

        A = set(last_partition_scopes)
        B = set(generated_last_stage_scopes)
        intersection = A & B
        correct = len(intersection)
        fp = len(A) - correct  # we predicted: true, result: false
        fn = len(B) - correct  # we predicted: false, result: true
        current_mistakes = fp + fn

        # stats:
        d = dict(correct=correct, fp=fp, fn=fn, mistakes=current_mistakes)

        history[n_runs] = dict(last_partition_scopes=last_partition_scopes,
                               generated_last_stage_scopes=generated_last_stage_scopes,
                               d=d
                               )
        # set current scopes as model scopes
        last_partition_scopes = generated_last_stage_scopes

        # log something
        print(f"run:{n_runs}", d)

    if not (current_mistakes > allowed_mistakes):
        print(f"Success! got {current_mistakes} mistakes after {n_runs} runs")
    elif not (n_runs_limit < 0 or n_runs < n_runs_limit):
        print(f"Breaking after reaching run limit of {n_runs_limit}!")
        i_min = list(history.keys())[np.argmin([v['d']['mistakes'] for v in history.values()])]
        mistakes_min = history[i_min]['d']['mistakes']
        print(f"Taking best seen: {mistakes_min} mistakes after {i_min} runs")
        print([history[i]['d']['mistakes'] for i in history])
        print(f"Restoring best point in history")
        # restore the best point from  history
        last_partition_scopes = history[i_min]['last_partition_scopes']
        for n in graph.nodes:
            if n.scope in last_partition_scopes:
                n.weight = weights[n.id].no_recomputation
            else:
                n.weight = weights[n.id].recomputation
        # Partition
        graph = partition_profiled_graph_fn(graph)

        # TODO: warn if not the same number...

    return graph
