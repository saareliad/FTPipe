import argparse
from argparse import Namespace
import sys
import os
import importlib
from typing import Dict, Optional, Tuple
sys.path.append("../")


from pytorch_Gpipe.model_profiling.control_flow_graph import Graph
from pytorch_Gpipe import pipe_model,get_weight_functions
from pytorch_Gpipe.utils import layerDict, tensorDict,move_tensors

from partitioning_scripts.partition_async_pipe import partition_async_pipe
from partitioning_scripts.partition_scripts_utils import bruteforce_main, choose_blocks, record_cmdline
from partitioning_scripts.tasks import Partitioner,get_parser_and_partitioner
from analysis import run_analysis
from analysis.analysis_utils import convert_to_analysis_format



def parse_cli()->Tuple[Namespace,Dict,Partitioner]:
    task_parser = argparse.ArgumentParser(description="partitioning task parser",add_help=False)
    task_parser.add_argument("--partitioning_task",help="partitioning task to perform")
    task,rest = task_parser.parse_known_args()

    parser_cls,partitioner_cls = get_parser_and_partitioner(task.partitioning_task)
    
    parser = parser_cls()

    cmd_args = parser.parse_args(args=rest)
    cmd_args.partitioning_task = task.partitioning_task

    model_args = dict()

    # group model args so we can save them in the generated file
    for group in parser._action_groups:
        if group.title == "model_args":
            model_args = {a.dest:getattr(cmd_args,a.dest,None) for a in group._group_actions}
            break

    return cmd_args,model_args,partitioner_cls(cmd_args)


def main(cmd_args:Namespace,model_args:Dict,partitioner:Partitioner,override_dict:Optional[Dict]=None):
    for i, v in override_dict.items():
        if i in model_args:
            raise ValueError(f"override dict should not modify model creation arguments got {i}\nthe intended use is for modifying partitioning/hueristics related values")
        setattr(cmd_args, i, v)

    model = partitioner.get_model(cmd_args).train()
    sample = partitioner.get_input(cmd_args, analysis=False)

    if isinstance(sample,dict):
        kwargs = sample
        args = tuple()
    elif isinstance(sample,tuple):
        kwargs = dict()
        args = sample
    else:
        kwargs = dict()
        args = (sample,)
    
    del sample

    if not cmd_args.save_memory_mode:
        model,args,kwargs = move_tensors((model,args,kwargs),cmd_args.device)
    

    node_weight_function, edge_weight_function = get_weight_functions(cmd_args, verbose=True)

    basic_blocks = choose_blocks(model, cmd_args)

    profiles_cache_name = cmd_args.profiles_cache_name
    overwrite_profiles_cache = cmd_args.overwrite_profiles_cache

    partitioner.register_functions()
    if cmd_args.async_pipeline and (not cmd_args.no_recomputation):
        print("-I- using async partitioner")
        graph = partition_async_pipe(cmd_args, model, batch_dim=partitioner.batch_dim,args=args, kwargs=kwargs,
                                        node_weight_function=node_weight_function,
                                        edge_weight_function=edge_weight_function)
    else:
        if profiles_cache_name and os.path.exists(profiles_cache_name) and not overwrite_profiles_cache:
            print(f"-V- loading profiles from cache: {profiles_cache_name}")
            graph = Graph.deserialize(profiles_cache_name)
        else:
            graph = None

        #apply partitioning either from scratch or from a cached graph
        graph = pipe_model(model,
                        partitioner.batch_dim,
                        basic_blocks=basic_blocks,
                        depth=cmd_args.depth,
                        args = args,
                        kwargs=kwargs,
                        nparts=cmd_args.n_partitions,
                        output_file=cmd_args.output_file,
                        generate_model_parallel=cmd_args.generate_model_parallel,
                        generate_explicit_del=cmd_args.generate_explicit_del,
                        use_layers_only_graph=True,
                        use_graph_profiler=not cmd_args.use_network_profiler,
                        use_network_profiler=cmd_args.use_network_profiler,
                        profile_ops=not cmd_args.disable_op_profiling,
                        node_weight_function=node_weight_function,
                        edge_weight_function=edge_weight_function,
                        n_iter=cmd_args.n_iter,
                        recomputation=not cmd_args.no_recomputation,
                        save_memory_mode=cmd_args.save_memory_mode,
                        use_METIS=cmd_args.use_METIS,
                        acyclic_opt=cmd_args.acyclic_opt,
                        METIS_opt=cmd_args.METIS_opt,
                        force_no_recomp_scopes=cmd_args.force_no_recomputation_scopes_fn,
                        graph=graph)

        # cache graph         
        if profiles_cache_name and not os.path.exists(profiles_cache_name):
            print(f"-V- writing to new cache: {profiles_cache_name}")
            graph.serialize(profiles_cache_name)
        elif profiles_cache_name and os.path.exists(profiles_cache_name) and overwrite_profiles_cache:
            print(f"-V- overwriting to cache: {profiles_cache_name}")
            graph.serialize(profiles_cache_name)

    del args,kwargs

    if cmd_args.dot:
        graph.save_as_pdf(cmd_args.output_file, ".")

    # Add cmdline to generate output file.
    record_cmdline(cmd_args.output_file)
    #record model creation args as a model_args variable
    with open(f"{cmd_args.output_file}.py", "a") as f:
        f.write("\n")
        f.write(f"model_args = {model_args}")

#record overriden args as a override_dict variable
    if override_dict:
        with open(f"{cmd_args.output_file}.py", "a") as f:
            f.write("\n")
            f.write(f"override_dict = {override_dict}")
        

    if not cmd_args.no_analysis:
        #load configuration for analysis
        module_path = cmd_args.output_file.replace("/", ".")
        generated = importlib.import_module(module_path)
        create_pipeline_configuration = generated.create_pipeline_configuration
        config = create_pipeline_configuration(DEBUG=True)
        layers = layerDict(model, depth=cmd_args.depth, basic_blocks=basic_blocks)
        tensors = tensorDict(model)
        analysis_config = convert_to_analysis_format(config,
                                                    layers,
                                                    tensors)
        del layers,tensors

        #run analysis log output in the generated file
        sample = partitioner.get_input(cmd_args, analysis=True)


        analysis_kwargs = dict(sample=sample,
                                graph=graph,
                                config=analysis_config,
                                n_iter=cmd_args.n_iter,
                                recomputation=not cmd_args.no_recomputation,
                                bw_GBps=cmd_args.bw,
                                verbose=True,
                                async_pipeline=cmd_args.async_pipeline,
                                sequential_model=model)
        
        analysis_kwargs = partitioner.update_analysis_kwargs(cmd_args,analysis_config,analysis_kwargs)
        analysis_result, summary = run_analysis(**analysis_kwargs)
        with open(f"{cmd_args.output_file}.py", "a") as f:
            f.write("\n")
            f.write('"""analysis summary\n' + summary + "\n" + '"""')
    else:
        analysis_result = summary = None


    partitioner.post_partitioning(cmd_args,graph,analysis_result,summary)

    return analysis_result,cmd_args.output_file

if __name__ == "__main__":
    cmd_args,model_args, partitioner = parse_cli() 

    #setup remote debugging
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
    
    NUM_RUNS = 2
    TMP = "tmp/"

    main_kwargs=dict(cmd_args=cmd_args,model_args=model_args,partitioner=partitioner)

    bruteforce_main(main,main_kwargs, override_dicts, NUM_RUNS, TMP)