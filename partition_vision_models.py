import torch
from models.normal import WideResNet, amoebanetd, vgg16_bn
from models.normal.vision_models import ResNet
from pytorch_Gpipe.utils import layerDict, tensorDict
from pytorch_Gpipe import PipelineConfig, pipe_model
import argparse
import importlib
from misc import run_analysis, run_partitions
import sys
from heuristics import edge_weight_function, node_weight_function
from partition_scripts_utils import ParseMetisOpts, ParsePartitioningOpts, record_cmdline, run_x_tries_until_no_fail
import functools
from partition_async_pipe import AsyncPipePartitioner

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


class ParsePartitioningOptsVision(ParsePartitioningOpts):
    def __init__(self,
                 model_choices=MODEL_CONFIGS.keys(),
                 dataset_choices=DATASETS):
        super().__init__()
        self.model_choices = model_choices
        self.dataset_choices = dataset_choices

    def _extra(self, parser):
        parser.add_argument('--model',
                            choices=self.model_choices,
                            default='wrn_16x4')
        parser.add_argument('-d',
                            '--dataset',
                            choices=self.dataset_choices,
                            default='cifar10')

    def set_defaults(self, parser):
        d = {
            # "model": 'wrn_16x4',
            "partitioning_batch_size": 128,
            "n_iter": 100,
            "output_file": 'wrn_16x4',
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 32,
        }

        parser.set_defaults(**d)


def parse_cli():

    parser = argparse.ArgumentParser(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ParsePartitioningOptsVision().add_partitioning_arguments(parser)
    ParseMetisOpts.add_metis_arguments(parser)

    args = parser.parse_args()
    args.auto_file_name = not args.no_auto_file_name
    if args.auto_file_name:
        args.output_file = f"{args.model}_p{args.n_partitions}"

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    METIS_opt = ParseMetisOpts.metis_opts_dict_from_parsed_args(args)
    return args, METIS_opt


if __name__ == "__main__":

    args, METIS_opt = parse_cli()
    GET_PARTITIONS_ON_CPU = True

    # if the model is too big run the whole partitioning process on CPU
    # and drink a cup of coffee in the meantime
    # define model and sample batch
    model = create_model(args.model)
    sample = create_random_sample(args, analysis=False)

    # TODO: combine the save_memory_mode with this...
    if args.model_too_big:
        model = model.cpu()
        sample = sample.cpu()
    else:
        if not args.save_memory_mode:
            # Will be sent to cuda when needed.
            model = model.cuda()
        sample = sample.cuda()

    # partition the model using our profiler
    # if the model need multiple inputs pass a tuple
    # if the model needs kwargs pass a dictionary
    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    n_partitions = args.n_partitions
    batch_dim = 0
    bwd_to_fwd_ratio = args.bwd_to_fwd_ratio

    partial_pipe_model = functools.partial(
        pipe_model,
        model,
        batch_dim,
        sample,
        depth=args.depth,
        kwargs=None,
        nparts=n_partitions,
        output_file=args.output_file,
        generate_model_parallel=args.generate_model_parallel,
        use_layers_only_graph=args.partition_layer_graph,
        node_weight_function=node_weight_function(
            bwd_to_fwd_ratio=bwd_to_fwd_ratio),
        edge_weight_function=edge_weight_function(
            bw, bwd_to_fwd_ratio=bwd_to_fwd_ratio),
        n_iter=n_iter,
        recomputation=recomputation,
        save_memory_mode=args.save_memory_mode,
        METIS_opt=METIS_opt)

    if args.async_pipeline and (not args.no_recomputation):
        async_pipe_partitioner = AsyncPipePartitioner(model, args.output_file,
                                                      partial_pipe_model)

        graph = run_x_tries_until_no_fail(
            async_pipe_partitioner.partition,
            10,
            # force_no_recomp_scopes=force_no_recomputation_fn,
            allowed_mistakes=0)
    else:
        graph = pipe_model(
            model,
            batch_dim,
            sample,
            depth=args.depth,
            kwargs=None,
            nparts=n_partitions,
            output_file=args.output_file,
            generate_model_parallel=args.generate_model_parallel,
            use_layers_only_graph=args.partition_layer_graph,
            node_weight_function=node_weight_function(
                bwd_to_fwd_ratio=bwd_to_fwd_ratio),
            edge_weight_function=edge_weight_function(
                bw, bwd_to_fwd_ratio=bwd_to_fwd_ratio),
            n_iter=n_iter,
            recomputation=recomputation,
            save_memory_mode=args.save_memory_mode,
            METIS_opt=METIS_opt)

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")
        graph.serialize(args.output_file)

    # Add cmdline to generate output file.
    record_cmdline(args.output_file)

    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration

    if GET_PARTITIONS_ON_CPU:
        sample = sample.to('cpu')
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

    pipe_config = PipelineConfig.fromDict(config)
    pipe_config.toJson(f"{args.output_file}.json")

    if not (args.no_test_run and args.no_analysis):
        depth = pipe_config.depth
        blocks = pipe_config.basic_blocks
        analysis_config = pipe_config._to_old_format(
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

    # Test # TODO: can do it on GPU...
    if not args.no_test_run:
        _ = run_partitions(sample, analysis_config)

    if not args.no_analysis:
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
