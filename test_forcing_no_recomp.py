from pytorch_Gpipe import pipe_model
from misc import run_analysis,convert_to_analysis_format
from models.normal import alexnet
import torch
from types import SimpleNamespace
import importlib
from partition_async_pipe import AsyncPipePartitioner
from functools import partial
from pytorch_Gpipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction
import os

from models.normal.NLP_models.stateless import StatelessLinear, StatelessEmbedding, StatelessSequential, CompositionStatelessSequential

# For debugging inside docker.
DEBUG = False
if DEBUG:
    import ptvsd
    port = 12345
    address = ('0.0.0.0', port)
    print(f"-I- waiting for attachment on {address}")
    ptvsd.enable_attach(address=address)
    ptvsd.wait_for_attach()
    breakpoint()

MODEL = 'SEQ'
MODEL = 'SEQ_TIED'
args = SimpleNamespace(output_file=f"results/generated_{MODEL}",
                       no_test_run=False,
                       no_analysis=False,
                       async_pipeline=True,
                       batch=256,
                       analyze_traced_model=False,
                       no_recomputation=False,
                       bwd_to_fwd_ratio=2,
                       n_partitions=2,
                       bw_GBps=12,
                       n_iter=1,
                       stateless_tied=False)

if not os.path.exists("results"):
    os.makedirs("result")

if MODEL == 'SEQ':
    model_dim = 10
    n_layers = 40
    model = torch.nn.Sequential(
        *[torch.nn.Linear(model_dim, model_dim) for i in range(n_layers)])
    model = model.cuda()
    sample = torch.randn(args.batch, model_dim).cuda()
elif MODEL == 'SEQ_TIED':
    model_dim = 1024
    n_layers = 8
    setattr(args, "stateless_tied", True)
    layers = [torch.nn.Linear(model_dim, model_dim) for i in range(n_layers)]
    model = CompositionStatelessSequential(*layers)
    # TODO: stages on same GPU...
    # TODO: currently we are not having it tied...
    model = model.cuda()
    sample = torch.randn(args.batch, model_dim).cuda()
elif MODEL == 'ALEXNET':
    model = alexnet().cuda()
    sample = torch.randn(args.batch, 3, 224, 224).cuda()
else:
    raise NotImplementedError()

#graph = pipe_model(model, 0, sample, n_iter=1, output_file=args.output_file)

nwf = NodeWeightFunction(bwd_to_fwd_ratio=args.bwd_to_fwd_ratio)
ewf = EdgeWeightFunction(args.bw_GBps,
                           bwd_to_fwd_ratio=args.bwd_to_fwd_ratio)

partial_pipe_model = partial(pipe_model,
                             model,
                             0,
                             sample,
                             n_iter=args.n_iter,
                             nparts=args.n_partitions,
                             output_file=args.output_file,
                             node_weight_function=nwf,
                             edge_weight_function=ewf,
                             recomputation=not args.no_recomputation)
if args.async_pipeline and (not args.no_recomputation):
    async_pipe_partitioner = AsyncPipePartitioner(model, args.output_file,
                                                  partial_pipe_model)

graph = async_pipe_partitioner.partition(allowed_mistakes=0)
graph.save_as_pdf(args.output_file,
                  ".",
                  node_weight_function=nwf,
                  edge_weight_function=ewf)
graph.serialize(args.output_file)

# Replace the dummy partition wtih cuda:0.
if args.stateless_tied:
    try:
        import subprocess
        subprocess.check_output([
            'sed', '-s', '-i', f"s/cuda:{args.n_partitions}/cuda:0/g",
            args.output_file + ".py"
        ])
    except:
        print("Failed to replaced tied dummy partition device")

module_path = args.output_file.replace("/", ".")
generated = importlib.import_module(module_path)
create_pipeline_configuration = generated.create_pipeline_configuration
layerDict = generated.layerDict
tensorDict = generated.tensorDict

GET_PARTITIONS_ON_CPU = True
config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)



depth = config['depth']
blocks = config['basic_blocks']
analysis_config = convert_to_analysis_format(config,
    layerDict(model, depth=depth, basic_blocks=blocks), tensorDict(model))

stages_on_same_gpu = set()
if args.stateless_tied and len(config['stages']) == args.n_partitions + 1:
    stages_on_same_gpu = [{0, args.n_partitions}]

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
    stages_on_same_gpu=stages_on_same_gpu)
