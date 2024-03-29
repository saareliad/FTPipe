import argparse
import importlib
import os
import sys
from argparse import Namespace
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch

from autopipe.analysis import run_analysis
from autopipe.analysis.analysis_utils import convert_to_analysis_format
from autopipe.autopipe import get_weight_functions
from autopipe.autopipe.api import pipe_model
from autopipe.autopipe.model_profiling.control_flow_graph import NodeTypes
from autopipe.autopipe.utils import layerDict, tensorDict, move_tensors
from autopipe.partitioning_scripts.partition_scripts_utils import bruteforce_main, choose_blocks, record_cmdline
from autopipe.tasks import PartitioningTask, Parser, get_parser_and_partitioner
from pipe.misc.sanity_check import run_sanity_check


def parse_cli() -> Tuple[Namespace, Dict, PartitioningTask]:
    task_parser = argparse.ArgumentParser(description="partitioning task parser", add_help=False)
    task_parser.add_argument("partitioning_task", help="partitioning task to perform")
    task, rest = task_parser.parse_known_args()

    parser_cls, partitioner_cls = get_parser_and_partitioner(task.partitioning_task)

    parser: Parser = parser_cls()

    cmd_args = parser.parse_args(args=rest)
    cmd_args.partitioning_task = task.partitioning_task

    model_args = dict()

    # group model args so we can save them in the generated file
    for group in parser._action_groups:
        if group.title == "model_args":
            model_args = {a.dest: getattr(cmd_args, a.dest, None) for a in group._group_actions}
            break

    return cmd_args, model_args, partitioner_cls(cmd_args)


def main(cmd_args: Namespace, model_args: Dict, partitioner: PartitioningTask, override_dict: Optional[Dict] = None):
    for i, v in override_dict.items():
        if i in model_args:
            raise ValueError(
                f"override dict should not modify model creation arguments got {i}\nthe intended use is for modifying partitioning/hueristics related values")
        setattr(cmd_args, i, v)

    if getattr(cmd_args, "sanity_check", False):
        torch.manual_seed(0)
        torch.cuda.synchronize()

    model = partitioner.get_model(cmd_args).train()
    sample = partitioner.get_input(cmd_args, analysis=False)

    if isinstance(sample, dict):
        kwargs = sample
        args = tuple()
    elif isinstance(sample, tuple):
        kwargs = dict()
        args = sample
    else:
        kwargs = dict()
        args = (sample,)

    del sample



    node_weight_function, edge_weight_function = get_weight_functions(cmd_args, verbose=True)


    profiles_cache_name = cmd_args.profiles_cache_name
    trace_cache_name = cmd_args.trace_cache_name
    overwrite_profiles_cache = cmd_args.overwrite_profiles_cache
    overwrite_trace_cache = cmd_args.overwrite_trace_cache

    partitioner.register_functions()
    if not cmd_args.analysis_only:
        if profiles_cache_name and os.path.exists(profiles_cache_name) and overwrite_profiles_cache:
            os.remove(profiles_cache_name)

        if trace_cache_name and os.path.exists(trace_cache_name) and overwrite_trace_cache:
            os.remove(trace_cache_name)

        if not cmd_args.save_memory_mode:
            with torch.no_grad():
                model, args, kwargs = move_tensors((model, args, kwargs), cmd_args.device)
        cmd_args.basic_blocks = choose_blocks(model, cmd_args, blocks_arg_name="basic_blocks")

        cmd_args.mpipe_opt['special_blocks'] = choose_blocks(model, cmd_args, blocks_arg_name="special_blocks")
        cmd_args.mpipe_opt['basic_blocks'] = cmd_args.basic_blocks

        # apply partitioning
        graph = pipe_model(model, partitioner.batch_dim, model_args=args, model_kwargs=kwargs,
                           n_iter=cmd_args.n_iter, nparts=cmd_args.n_partitions, depth=cmd_args.depth,
                           basic_blocks=cmd_args.basic_blocks, node_weight_function=node_weight_function,
                           edge_weight_function=edge_weight_function, use_layers_only_graph=True,
                           output_file=cmd_args.output_file,
                           generate_explicit_del=cmd_args.generate_explicit_del,
                           generate_activation_propagation=not cmd_args.no_activation_propagation,
                           recomputation=not cmd_args.no_recomputation,
                           partitioning_method=cmd_args.partitioning_method,
                           METIS_opt=cmd_args.METIS_opt,
                           acyclic_opt=cmd_args.acyclic_opt,
                           binpack_opt=cmd_args.binpack_opt,
                           mpipe_opt=cmd_args.mpipe_opt,
                           force_no_recomp_scopes=cmd_args.force_no_recomputation_scopes_fn,
                           save_memory_mode=cmd_args.save_memory_mode,
                           trace_on_gpu=cmd_args.trace_on_gpu,
                           use_graph_profiler=not cmd_args.use_network_profiler,
                           use_network_profiler=cmd_args.use_network_profiler,
                           profile_ops=not cmd_args.disable_op_profiling,
                           graph=None,  # TODO: deprecated
                           async_pipe=cmd_args.async_pipeline,
                           trace_cache_name=trace_cache_name,
                           profiles_cache_name=profiles_cache_name,
                           dont_use_async_meta_alg=cmd_args.dont_use_async_meta_alg)

        del args, kwargs

        if cmd_args.dot:
            try:
                graph.save_as_pdf(cmd_args.output_file, "graphs",
                                  node_weight_function=node_weight_function, edge_weight_function=edge_weight_function)
            except Exception as e:
                print("Error saving graph as pdf")
                raise e

        # Add cmdline to generate output file.
        record_cmdline(cmd_args.output_file)
        # record model creation args as a model_args variable
        with open(f"{cmd_args.output_file}.py", "a") as f:
            f.write("\n")
            f.write(f"model_args = {model_args}")

        if hasattr(partitioner, "record_transformer_cfg"):
            partitioner.record_transformer_cfg(cmd_args=cmd_args)

        # record overridden args as a override_dict variable
        if override_dict:
            with open(f"{cmd_args.output_file}.py", "a") as f:
                f.write("\n")
                f.write(f"override_dict = {override_dict}")
    else:
        graph = None

    if not cmd_args.no_analysis:
        # load configuration for analysis
        if sys.platform == "win32":
            module_path = cmd_args.output_file.replace('\\', ".")
        else:
            module_path = cmd_args.output_file.replace("/", ".")

        generated = importlib.import_module(module_path)
        create_pipeline_configuration = generated.create_pipeline_configuration
        config = create_pipeline_configuration(DEBUG=True)
        layers = layerDict(model, depth=config['depth'], basic_blocks=config['basic_blocks'])
        tensors = tensorDict(model)
        analysis_config = convert_to_analysis_format(config,
                                                     layers,
                                                     tensors)
        del layers, tensors

        if getattr(cmd_args, "sanity_check", False):
            print("-I- running sanity check")
            run_sanity_check(cmd_args, partitioner, analysis_config, device=cmd_args.device,
                             training=True, check_grads=True, ref_model=None, check_init=False)

        # run analysis log output in the generated file
        sample = partitioner.get_input(cmd_args, analysis=True)

        analysis_kwargs = dict(sample=sample,
                               graph=graph,
                               config=analysis_config,
                               n_iter=cmd_args.n_iter,
                               recomputation=not cmd_args.no_recomputation,
                               bw_GBps=cmd_args.bw,
                               verbose=True,
                               async_pipeline=cmd_args.async_pipeline or cmd_args.analysis_as_async_pipeline,
                               sequential_model=model)

        if cmd_args.partitioning_method != 'ACYCLIC':
            gpu_to_stages = defaultdict(set)
            stage_to_gpu = dict()
            for n in graph.non_input_nodes:
                if n.gpu_id is None or n.type == NodeTypes.CONSTANT:
                    continue
                gpu_to_stages[n.gpu_id].add(n.stage_id)
                if n.stage_id in stage_to_gpu:
                    assert stage_to_gpu[n.stage_id] == n.gpu_id, (stage_to_gpu[n.stage_id], n.gpu_id)
                else:
                    assert n.gpu_id is not None
                stage_to_gpu[n.stage_id] = n.gpu_id
            if gpu_to_stages:
                analysis_kwargs['stages_on_same_gpu'] = list(gpu_to_stages.values())
            stage_to_gpu = [stage_to_gpu[i] for i in sorted(stage_to_gpu.keys())]
            print("stage_to_gpu", stage_to_gpu)

        analysis_kwargs = partitioner.update_analysis_kwargs(cmd_args, analysis_config, analysis_kwargs)
        analysis_result, summary = run_analysis(**analysis_kwargs)
        with open(f"{cmd_args.output_file}.py", "a") as f:
            f.write("\n")
            f.write('"""analysis summary\n' + summary + "\n" + '"""')
    else:
        analysis_result = summary = None

    partitioner.post_partitioning(cmd_args, graph, analysis_result, summary)

    return analysis_result, cmd_args.output_file


if __name__ == "__main__":
    cmd_args, model_args, partitioner = parse_cli()
    # TODO we can cache the models when doing hyper parameter searching
    # setup remote debugging
    if cmd_args.debug:
        import ptvsd

        address = ('127.0.0.1', 3000)
        print(f"-I- rank waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()
        print("attached")

    override_dicts = []  # list of dicts to override args with...

    # TODO: put all hyper parameters here, a dict for each setting we want to try.
    # d1 = dict(basic_blocks=["T5LayerCrossAttention","T5LayerSelfAttention"])
    # override_dicts.append(d1)

    NUM_RUNS = 1
    TMP = "tmp/"
    output_dir = os.path.join("models", "partitioned")

    cmd_args.output_file = os.path.join(output_dir, cmd_args.output_file)

    main_kwargs = dict(cmd_args=cmd_args, model_args=model_args, partitioner=partitioner)

    bruteforce_main(main, main_kwargs, override_dicts, NUM_RUNS, TMP)
