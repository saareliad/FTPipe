from pytorch_Gpipe import pipe_model, PipelineConfig
from misc import run_analysis
from models.normal import alexnet
import torch
from types import SimpleNamespace
import importlib
from partition_async_pipe import AsyncPipePartitioner
from functools import partial
from heuristics import node_weight_function, edge_weight_function

# For debugging inside docker.
import ptvsd
DEBUG = False
if DEBUG:
    port = 12345
    address = ('0.0.0.0', port)
    print(f"-I- waiting for attachment on {address}")
    ptvsd.enable_attach(address=address)
    ptvsd.wait_for_attach()
    breakpoint()

MODEL = 'SEQ'
args = SimpleNamespace(output_file=f"generated_{MODEL}",
                       no_test_run=False,
                       no_analysis=False,
                       async_pipeline=True,
                       batch=256,
                       analyze_traced_model=True,
                       no_recomputation=False,
                       bwd_to_fwd_ratio=-1,
                       n_partitions=2,
                       bw_GBps=12,
                       n_iter=1)
if MODEL == 'SEQ':
    model_dim = 10
    model = torch.nn.Sequential(*[torch.nn.Linear(model_dim, model_dim) for i in range(40)])
    model = model.cuda()
    sample = torch.randn(args.batch, model_dim).cuda()
elif MODEL == 'ALEXNET':
    model = alexnet().cuda()
    sample = torch.randn(args.batch, 3, 224, 224).cuda()
else:
    raise NotImplementedError()

#graph = pipe_model(model, 0, sample, n_iter=1, output_file=args.output_file)
partial_pipe_model = partial(pipe_model,
                             model,
                             0,
                             sample,
                             n_iter=args.n_iter,
                             nparts=args.n_partitions,
                             output_file=args.output_file,
                             node_weight_function=node_weight_function(
                                 bwd_to_fwd_ratio=args.bwd_to_fwd_ratio),
                             edge_weight_function=edge_weight_function(
                                 args.bw_GBps,
                                 bwd_to_fwd_ratio=args.bwd_to_fwd_ratio),
                             recomputation=not args.no_recomputation)
if args.async_pipeline and (not args.no_recomputation):
    async_pipe_partitioner = AsyncPipePartitioner(model, args.output_file,
                                                  partial_pipe_model)

graph = async_pipe_partitioner.partition(allowed_mistakes=0)

module_path = args.output_file.replace("/", ".")
generated = importlib.import_module(module_path)
create_pipeline_configuration = generated.create_pipeline_configuration
layerDict = generated.layerDict
tensorDict = generated.tensorDict

GET_PARTITIONS_ON_CPU = True
if GET_PARTITIONS_ON_CPU:
    # TODO: not sure about this...
    sample = sample.to('cpu')
config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

pipe_config = PipelineConfig.fromDict(config)
pipe_config.toJson(f"{args.output_file}.json")

depth = pipe_config.depth
blocks = pipe_config.basic_blocks
analysis_config = pipe_config._to_old_format(
    layerDict(model, depth=depth, basic_blocks=blocks), tensorDict(model))

analysis_result, summary = run_analysis(
    sample,
    graph,
    analysis_config,
    n_iter=args.n_iter,
    recomputation=not args.no_recomputation,
    bw_GBps=args.bw_GBps,
    verbose=True,
    async_pipeline=args.async_pipeline,
    sequential_model=model,
    analyze_traced_model=args.analyze_traced_model)
