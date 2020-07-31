import sys
sys.path.append("../")
import torch
from models.normal import WideResNet, amoebanetd, vgg16_bn
from models.normal.vision_models import ResNet
from pytorch_Gpipe.utils import layerDict, tensorDict
from pytorch_Gpipe import pipe_model
import argparse
import importlib
from analysis import run_analysis,convert_to_analysis_format
from pytorch_Gpipe import get_weight_functions
from partition_scripts_utils import Parser, record_cmdline, choose_blocks
import functools
from partition_async_pipe import partition_async_pipe

_VGG16_BN = dict(vgg16_bn=dict())

_RESENETS = dict(resnet50_imagenet=dict(block=ResNet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        num_classes=1000),
                 resnet101_imagenet=dict(block=ResNet.Bottleneck,
                                         layers=[3, 4, 23, 3],
                                         num_classes=1000))

_WIDE_RESNETS = dict(
    wrn_16x4=dict(depth=16, num_classes=10, widen_factor=4,
                  drop_rate=0.0),  # FOR BACKWARD COMPATABILITY
    wrn_16x4_c10=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    wrn_16x4_c100=dict(depth=16,
                       num_classes=100,
                       widen_factor=4,
                       drop_rate=0.0),
    wrn_28x10_c10_dr03=dict(depth=28,
                            num_classes=10,
                            widen_factor=10,
                            drop_rate=0.3),
    wrn_28x10_c10=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0),
    wrn_28x10_c100_dr03=dict(depth=28,
                             num_classes=100,
                             widen_factor=10,
                             drop_rate=0.3),
    wrn_28x10_c100=dict(depth=28,
                        num_classes=100,
                        widen_factor=10,
                        drop_rate=0),
)

# this model is realy big even with 4 cells it contains 845 layers
_AMOEBANET_D = dict(amoebanet_4x512_c10=dict(num_layers=4,
                                             num_filters=512,
                                             num_classes=10),
                    amoebanet_8x512_c100=dict(num_layers=8,
                                              num_filters=512,
                                              num_classes=100))

MODEL_CFG_TO_SAMPLE_MODEL = {}
MODEL_CONFIGS = {}
# CFG_TO_GENERATED_FILE_NAME = {}


def _register_model(dict_params, model_cls):
    global MODEL_CFG_TO_SAMPLE_MODEL
    global MODEL_CONFIGS
    # global CFG_TO_GENERATED_FILE_NAME

    MODEL_CONFIGS.update(dict_params)
    MODEL_CFG_TO_SAMPLE_MODEL.update(
        {k: model_cls
         for k in dict_params.keys()})

    # CFG_TO_GENERATED_FILE_NAME = {i: i for i in MODEL_CONFIGS.keys()}


_register_model(_WIDE_RESNETS, WideResNet)
_register_model(_RESENETS, ResNet.ResNet)
_register_model(_AMOEBANET_D, amoebanetd)
_register_model(_VGG16_BN, vgg16_bn)

DATASETS = ['cifar10', 'cifar100', 'imagenet']


def create_model(cfg='wrn_16x4'):
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg](**MODEL_CONFIGS[cfg])


# TODO: option to be more accurate if we use real data,
# taking stuff like sparsity in considuration.
def create_random_sample(args, analysis=False):
    dataset = args.dataset
    if analysis:
        batch_size = args.analysis_batch_size
    else:
        batch_size = args.partitioning_batch_size

    if dataset == 'cifar10' or dataset == 'cifar100':
        sample = torch.randn(batch_size, 3, 32, 32)
    elif dataset == 'imagenet':
        sample = torch.randn(batch_size, 3, 224, 224)

    return sample


class ParsePartitioningOptsVision(Parser):
    def _add_model_args(self,group):
        group.add_argument('--model',
                            choices=MODEL_CONFIGS.keys(),
                            default='wrn_16x4')
                            
    def _add_data_args(self, group):
        group.add_argument('-d',
                            '--dataset',
                            choices=DATASETS,
                            default='cifar10')

    def _default_values(self):
        return {
            # "model": 'wrn_16x4',
            "partitioning_batch_size": 128,
            "n_iter": 100,
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 32,
        }



def parse_cli():
    parser = ParsePartitioningOptsVision(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()
    if not args.output_file:
        args.output_file = f"{args.model}_p{args.n_partitions}"

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    return args


if __name__ == "__main__":
    args = parse_cli()
    GET_PARTITIONS_ON_CPU = True

    # if the model is too big run the whole partitioning process on CPU
    # and drink a cup of coffee in the meantime
    # define model and sample batch
    model = create_model(args.model)
    sample = create_random_sample(args, analysis=False)

    if not args.save_memory_mode:
        model = model.to(args.device)
        sample = sample.to(args.device)

    # partition the model using our profiler
    # if the model need multiple inputs pass a tuple
    # if the model needs kwargs pass a dictionary
    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    n_partitions = args.n_partitions
    batch_dim = 0
    bwd_to_fwd_ratio = args.bwd_to_fwd_ratio
    args.basic_blocks = choose_blocks(model, args)

    node_weight_function, edge_weight_function = get_weight_functions(args, verbose=True)

    partial_pipe_model = functools.partial(
        pipe_model,
        model,
        batch_dim,
        sample,
        basic_blocks=args.basic_blocks,
        depth=args.depth,
        kwargs=None,
        nparts=n_partitions,
        output_file=args.output_file,
        generate_model_parallel=args.generate_model_parallel,
        generate_explicit_del=args.generate_explicit_del,
        use_layers_only_graph=True,  # FIXME:
        use_graph_profiler=not args.use_network_profiler,
        use_network_profiler=args.use_network_profiler,
        profile_ops=not args.disable_op_profiling,
        node_weight_function=node_weight_function,
        edge_weight_function=edge_weight_function,
        n_iter=n_iter,
        recomputation=recomputation,
        save_memory_mode=args.save_memory_mode,
        use_METIS=args.use_METIS,
        acyclic_opt=args.acyclic_opt,
        METIS_opt=args.METIS_opt)

    if args.async_pipeline and (not args.no_recomputation):
        print("using async partitioner")
        graph = partition_async_pipe(args, model, 0, sample,
                                    node_weight_function=node_weight_function,
                                    edge_weight_function=edge_weight_function,)
    else:
        graph = partial_pipe_model()

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")
        graph.serialize(args.output_file)

    # Add cmdline to generate output file.
    record_cmdline(args.output_file)

    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration

    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)


    if not args.no_analysis:
        depth = args.depth
        blocks = args.basic_blocks
        analysis_config = convert_to_analysis_format(config,
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

        sample = create_random_sample(args, analysis=True)
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