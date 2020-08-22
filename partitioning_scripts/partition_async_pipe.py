import sys
sys.path.append("../")
import torch
from collections import namedtuple
from typing import Dict, Optional
import functools
import pickle
import os

from pytorch_Gpipe import trace_module, Graph, GraphProfiler, execute_graph, ExecTimes, acyclic_partition, infer_req_grad, compile_partitioned_model, METIS_partition, profile_network
from pytorch_Gpipe.model_profiling import Node
from pytorch_Gpipe.utils import move_tensors

FullExecTimes = namedtuple('FullExecTimes', 'recomputation no_recomputation')


def partition_async_pipe(
    cmd_args,
    model,
    batch_dim: int = 0,
    args: tuple = None,
    kwargs: Dict = None,
    node_weight_function=None,
    edge_weight_function=None,
    # CACHE
    profiles_cache_name="",
    overwrite_profiles_cache=False,
)->Graph:

    allowed_mistakes = 0
    # HACK: allow mistakes for multilevel and acyclic...
    if not cmd_args.use_METIS:
        allowed_mistakes += 2

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    # combined node/edge weight function depends on how many parameters are passed
    evaluator = Evaluator(node_weight_function, edge_weight_function)

    profiles_cache_name = getattr(cmd_args, "profiles_cache_name", None)
    overwrite_profiles_cache = getattr(cmd_args, "overwrite_profiles_cache", False)

    if profiles_cache_name and os.path.exists(profiles_cache_name) and not overwrite_profiles_cache:
        print(f"-V- loading profiles from cache: {profiles_cache_name}")
        with open(profiles_cache_name, "rb") as f:
            graph, weights = pickle.load(f)
    else:
        graph = trace_module(model,
                             args=args,
                             kwargs=kwargs,
                             depth=cmd_args.depth,
                             basic_blocks=cmd_args.basic_blocks)

        weights = full_profile(graph, model, args, kwargs, cmd_args)

        if profiles_cache_name and os.path.exists(profiles_cache_name):
            if not overwrite_profiles_cache:
                print("-V- will not overrite cache with current graph and weights")
            else:
                print(f"-V- overwriting to cache: {profiles_cache_name}")
                with open(profiles_cache_name, "wb") as f:
                    pickle.dump((graph, weights), f)
        elif profiles_cache_name:
            print(f"-V- writing to new cache: {profiles_cache_name}")
            with open(profiles_cache_name, "wb") as f:
                pickle.dump((graph, weights), f)

    last_partition_scopes = set()

    current_mistakes = allowed_mistakes + 1
    n_runs = 0

    while current_mistakes > allowed_mistakes:

        for n in graph.nodes:
            if n.scope in last_partition_scopes:
                n.weight = weights[n].no_recomputation
            else:
                n.weight = weights[n].recomputation

        if not cmd_args.use_METIS:
            acyclic_partition(graph,
                              cmd_args.n_partitions,
                              node_weight_function=evaluator,
                              edge_weight_function=evaluator,
                              use_layers_graph=True,
                              **cmd_args.acyclic_opt)
        else:
            METIS_partition(graph,
                            cmd_args.n_partitions,
                            node_weight_function=evaluator,
                            edge_weight_function=evaluator,
                            use_layers_graph=True,
                            **cmd_args.METIS_opt)
        n_runs += 1

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
        # set current scopes as model scopes
        last_partition_scopes = generated_last_stage_scopes

        # log something
        print(f"run:{n_runs}", d)

    print(f"Success! got {current_mistakes} mistakes after {n_runs} runs")

    infer_req_grad(graph, model, args=args, kwargs=kwargs)

    compile_partitioned_model(
        graph,
        model,
        batch_dim,
        generate_explicit_del=cmd_args.generate_explicit_del,
        generate_model_parallel=cmd_args.generate_model_parallel,
        generate_activation_propagation=not cmd_args.no_activation_propagation,
        output_file=cmd_args.output_file)

    return graph


def full_profile(graph: Graph, model: torch.nn.Module, args: tuple,
                 kwargs: dict, cmd_args) -> Dict[Node, FullExecTimes]:
    if cmd_args.use_network_profiler:
        assert cmd_args.disable_op_profiling, "op profiling is not supported in the network profiler"
        profile_func = functools.partial(
            profile_network,
            model,
            args,
            kwargs,
            basic_blocks=cmd_args.basic_blocks,
            max_depth=cmd_args.depth,
            n_iter=cmd_args.n_iter,
            save_memory_mode=cmd_args.save_memory_mode,
            force_no_recomp_scopes=None)
    else:
        print(
            f"using graph profiler with op profiling = {not cmd_args.disable_op_profiling} save_memory_mode = {cmd_args.save_memory_mode}"
        )

        def profile_func(recomputation=True):
            profiler = GraphProfiler(
                recomputation=recomputation,
                n_iter=cmd_args.n_iter,
                profile_ops=not cmd_args.disable_op_profiling,
                force_no_recomp_scopes=None,
                save_memory_mode = cmd_args.save_memory_mode,
                )
            execute_graph(model,
                          graph,
                          model_args=args,
                          model_kwargs=kwargs,
                          pre_hook=profiler.time_forward,
                          post_hook=profiler.time_backward,
                          enforce_out_of_place=True)
            return profiler.get_weights()

    # profile recomputation
    if cmd_args.save_memory_mode:
        model, args, kwargs = move_tensors((model, args, kwargs), 'cpu')

    recomputation_times = profile_func(recomputation=True)
    for n in graph.nodes:
        if n.scope not in recomputation_times:
            recomputation_times[n.scope] = ExecTimes(0, 0)

    # profile no recomputation
    if cmd_args.save_memory_mode:
        model, args, kwargs = move_tensors((model, args, kwargs), 'cpu')

    no_recomputation_times = profile_func(recomputation=False)
    for n in graph.nodes:
        if n.scope not in no_recomputation_times:
            no_recomputation_times[n.scope] = ExecTimes(0, 0)

    return {
        n: FullExecTimes(recomputation_times[n.scope],
                         no_recomputation_times[n.scope])
        for n in graph.nodes
    }


class Evaluator():
    def __init__(self, node_weight_function, edge_weight_function):
        self.node_evaluator = node_weight_function
        self.edge_evaluator = edge_weight_function

    def __call__(self, u: Node, v: Optional[Node] = None) -> float:
        if v is None:
            return self.node_evaluator(u)
        return self.edge_evaluator(u, v)
