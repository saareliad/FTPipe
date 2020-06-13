from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration, T5Model
from models.normal.NLP_models.modeling_T5_tied_weights import T5ForConditionalGeneration as TiedT5ForConditionalGeneration, T5Model as TiedT5Model
from transformers import T5Tokenizer,T5Config,T5_PRETRAINED_MODEL_ARCHIVE_MAP
import torch
import operator
import math
from pytorch_Gpipe import PipelineConfig,pipe_model
from pytorch_Gpipe.model_profiling.tracer import register_new_explicit_untraced_function,register_new_traced_function
from pytorch_Gpipe.utils import layerDict,tensorDict
import argparse
import importlib
from heuristics import NodeWeightFunction,EdgeWeightFunction
import functools
from partition_async_pipe import partition_async_pipe
from partition_scripts_utils import choose_blocks,ParseAcyclicPartitionerOpts,ParseMetisOpts,ParsePartitioningOpts,record_cmdline
from misc import run_analysis,run_partitions


def register_functions():
    register_new_explicit_untraced_function(operator.is_,operator)
    register_new_explicit_untraced_function(operator.is_not,operator)
    register_new_traced_function(math.log,math)
    register_new_traced_function(torch.einsum,torch)

def get_model_and_tokenizer(args):
    config = T5Config.from_pretrained(args.model)
    config.output_attentions=False
    config.output_hidden_states=False
    setattr(config,"output_only",True)
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    base = not args.lmhead
    tied = args.stateless_tied

    if base and tied:
        model_cls = TiedT5Model
    elif base and not tied:
        model_cls = T5Model
    elif not base and tied:
        model_cls = TiedT5ForConditionalGeneration
    else:
        model_cls = T5ForConditionalGeneration
    model = model_cls.from_pretrained(args.model,config=config).to(args.device).train()

    if tied:
        model.make_stateless()
    
    return model,tokenizer

def get_input(args,tokenizer,analysis=False):
    input_ids = tokenizer.encode(
        "Hello, my dog is cute", return_tensors="pt").to(args.device)  # Batch (1,6)
    
    if analysis:
        input_ids = input_ids.repeat(args.analysis_batch_size,10).contiguous() #Batch (ab,60)
    else:
        input_ids = input_ids.repeat(args.partitioning_batch_size,10).contiguous() #Batch (pb,60)
    
    if args.lmhead:
        kwargs = {"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids,"use_cache":False}
    else:
        kwargs = {"input_ids":input_ids,"decoder_input_ids":input_ids,"use_cache":False}
    
    return kwargs



class ParsePartitioningT5Opts(ParsePartitioningOpts):
    def _extra(self,parser):
        parser.add_argument("--model",choices=T5_PRETRAINED_MODEL_ARCHIVE_MAP.keys(),
                            default='t5-small')
        parser.add_argument("--stateless_tied",action="store_true",default=False)
        parser.add_argument("--lmhead",action="store_true",default=False)
    
    def set_defaults(self,parser):
        d = {
            "model":"t5-small",
            "partitioning_batch_size": 64,
            "n_iter": 50,
            "output_file": 'T5_small',
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 64,
            "basic_blocks": ["T5Attention"]
        }
        parser.set_defaults(**d)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ParsePartitioningT5Opts().add_partitioning_arguments(parser)
    ParseMetisOpts.add_metis_arguments(parser)
    ParseAcyclicPartitionerOpts.add_acyclic_partitioner_arguments(parser)

    args = parser.parse_args()
    args.auto_file_name = not args.no_auto_file_name
    if args.auto_file_name:
        args.output_file = f"{args.model.replace('-','_')}_p{args.n_partitions}"

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    args.METIS_opt = ParseMetisOpts.metis_opts_dict_from_parsed_args(args)
    args.acyclic_opt = ParseAcyclicPartitionerOpts.acyclic_opts_dict_from_parsed_args(args)

    
    device = "cuda" if (torch.cuda.is_available() and (not args.model_too_big)) else "cpu"
    device = torch.device(device)
    args.device = device

    return args



if __name__ == "__main__":
    args= parse_cli()

    model,tokenizer = get_model_and_tokenizer(args)

    sample = get_input(args,tokenizer,analysis=False)

    register_functions()
    # partition the model using our profiler
    # if the model need multiple inputs pass a tuple
    # if the model needs kwargs pass a dictionary
    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    n_partitions = args.n_partitions
    batch_dim = 0
    bwd_to_fwd_ratio = args.bwd_to_fwd_ratio
    args.basic_blocks = choose_blocks(model,args)
    partial_pipe_model = functools.partial(
        pipe_model,
        model,
        batch_dim,
        basic_blocks = args.basic_blocks,
        depth=args.depth,
        kwargs=sample,
        nparts=n_partitions,
        output_file=args.output_file,
        generate_model_parallel=args.generate_model_parallel,
        generate_explicit_del=args.generate_explicit_del,
        use_layers_only_graph=True,
        use_graph_profiler=not args.use_network_profiler,
        use_network_profiler=args.use_network_profiler,
        profile_ops=not args.disable_op_profiling,
        node_weight_function=NodeWeightFunction(
            bwd_to_fwd_ratio=bwd_to_fwd_ratio),
        edge_weight_function=EdgeWeightFunction(
            bw, bwd_to_fwd_ratio=bwd_to_fwd_ratio),
        n_iter=n_iter,
        recomputation=recomputation,
        save_memory_mode=args.save_memory_mode,
        use_METIS= args.use_METIS,
        acyclic_opt=args.acyclic_opt,
        METIS_opt=args.METIS_opt)

    if args.async_pipeline and (not args.no_recomputation):
        print("using async partitioner")
        graph = partition_async_pipe(args,model,0,kwargs=sample)
    else:
        graph = partial_pipe_model()

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")

    # Add cmdline to generate output file.
    record_cmdline(args.output_file)

    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration


    config = create_pipeline_configuration(DEBUG=True)

    pipe_config = PipelineConfig.fromDict(config)

    if not (args.no_test_run and args.no_analysis):
        depth = pipe_config.depth
        blocks = pipe_config.basic_blocks
        analysis_config = pipe_config._to_old_format(
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))
    
    if not args.no_test_run:
        _ = run_partitions(sample, analysis_config)

    if not args.no_analysis:
        sample = get_input(args,tokenizer, analysis=True)
        analysis_result, summary = run_analysis(
            sample,
            graph,
            analysis_config,
            n_iter,
            recomputation=recomputation,
            bw_GBps=bw,
            verbose=True,
            async_pipeline=args.async_pipeline,
            sequential_model=model)
        with open(f"{args.output_file}.py", "a") as f:
            f.write("\n")
            f.write('"""analysis summary\n' + summary + "\n" + '"""')