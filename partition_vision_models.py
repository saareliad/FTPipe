import torch
from models.normal import WideResNet, amoebanetd, ResNet, vgg16_bn
from pytorch_Gpipe import pipe_model
from pytorch_Gpipe.model_profiling import Node, NodeTypes
import argparse
import importlib
from misc import run_analysis, run_partitions

_VGG16_BN = dict(vgg16_bn=dict())

_RESENETS = dict(resnet50_imagenet=dict(
    block=ResNet.Bottleneck, layers=[3, 4, 6, 3], num_classes=1000))

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
        batch_size = args.batch_size

    if dataset == 'cifar10' or dataset == 'cifar100':
        sample = torch.randn(batch_size, 3, 32, 32)
    elif dataset == 'imagenet':
        sample = torch.randn(batch_size, 3, 224, 224)

    return sample


MULT_FACTOR = 1000


def node_weight_function(node: Node):
    # TODO: factory with recomputation.
    if node.type is NodeTypes.LAYER:
        return int(MULT_FACTOR *
                   (node.weight.backward_time))  # + node.weight.forward_time
    if node.type is NodeTypes.CONSTANT:
        return 0
    if node.type is NodeTypes.OP:  # FIXME:
        return 0
    return 0


def edge_weight_function(bw_GBps):
    def f(u: Node, v: Node):
        if u.type is NodeTypes.LAYER:
            return max(1, int(MULT_FACTOR * u.weight.output_size / bw_GBps))
        if v.type is NodeTypes.LAYER:
            return max(1, int(MULT_FACTOR * v.weight.input_size / bw_GBps))
        if u.type is NodeTypes.CONSTANT:
            return 1000 * MULT_FACTOR  # FIXME: why penalize constants?
        if v.type is NodeTypes.PYTHON_PRIMITIVE and v.valueType() in [list, tuple]:
            if "prim::TupleUnpack" in v.scope or "prim::ListUnpack" in v.scope:
                return 1000 * MULT_FACTOR
        if u.type is NodeTypes.PYTHON_PRIMITIVE and u.valueType() in [list, tuple]:
            if "prim::ListConstruct" in u.scope or "prim::TupleConstruct" in u.scope:
                return 1000 * MULT_FACTOR
        return 1

    return f


def parse_cli():

    parser = argparse.ArgumentParser(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
                        default='wrn_16x4',
                        choices=MODEL_CONFIGS.keys())
    parser.add_argument('-d', '--dataset', default='cifar10', choices=DATASETS)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument(
        '--model_too_big',
        action='store_true',
        default=False,
        help="if the model is too big run the whole partitioning process on CPU, "
        "and drink a cup of coffee in the meantime")
    parser.add_argument('-p', '--n_partitions', type=int, default=4)
    parser.add_argument('-o', '--output_file', default='wrn_16x4')
    parser.add_argument('--no_auto_file_name',
                        action='store_true',
                        default=False,
                        help="do not create file name automatically")
    parser.add_argument(
        '--n_iter',
        type=int,
        default=100,
        help="number of iteration used in order to profile the network and run analysis"
    )
    parser.add_argument(
        '--bw',
        type=float,
        default=12,
        help="data transfer rate between gpus in GBps (Gigabytes per second)")
    parser.add_argument(
        '--no_recomputation',
        action='store_true',
        default=False,
        help="whether to (not) use recomputation for the backward pass")
    parser.add_argument('--no_analysis',
                        action='store_true',
                        default=False,
                        help="disable partition analysis")
    parser.add_argument("--depth",
                        default=-1,
                        type=int,
                        help="the depth in which we will partition the model")
    parser.add_argument(
        "--partition_layer_graph",
        action="store_true",
        default=False,
        help="whether to partition a graph containing only layers")
    parser.add_argument(
        "--analysis_batch_size",
        default=32,
        type=int,
        help="batch size to use during the post partition analysis")
    parser.add_argument("-a",
                        "--async_pipeline",
                        default=False,
                        action="store_true",
                        help="Do analysis for async pipeline")
    parser.add_argument("--dot",
                        default=False,
                        action="store_true",
                        help="Save and plot it using graphviz")

    parser.add_argument("--no_test_run",
                        default=False,
                        action="store_true",
                        help="Do not try to run partitions after done")

    parser.add_argument(
        "--save_memory_mode",
        default=False,
        action="store_true",
        help="Save memory during profiling by storing everything on cpu," +
        " but sending each layer to GPU before the profiling.")

    metis_opts = parser.add_argument_group("METIS options")
    metis_opts.add_argument("--seed",
                            required=False,
                            type=int,
                            help="Random seed for Metis algorithm")
    metis_opts.add_argument(
        '--compress', default=False, action='store_true',
        help="Compress")  # NOTE: this is differnt from default!
    metis_opts.add_argument(
        '--metis_niter',
        type=int,
        help="Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process."
        "Default is 10.")
    metis_opts.add_argument(
        '--nseps',
        type=int,
        help="Specifies the number of different separators that it will compute at each level of nested dissection."
        "The final separator that is used is the smallest one. Default is 1.")
    metis_opts.add_argument(
        "--ncuts",
        type=int,
        help="Specifies the number of different partitionings that it will compute."
        " The final partitioning is the one that achieves the best edgecut or communication volume."
        "Default is 1.")
    metis_opts.add_argument(
        '--metis_dbglvl',
        type=int,
        help="Metis debug level. Refer to the docs for explanation")
    metis_opts.add_argument(
        '--objtype',
        type=int,
        help="Extra objective type to miminize (0: edgecut, 1: vol, default: edgecut)")

    args = parser.parse_args()
    args.auto_file_name = not args.no_auto_file_name
    if args.auto_file_name:
        args.output_file = f"{args.model}_p{args.n_partitions}"

    # TODO: build metis options
    # We can set to None to get the default
    # See See : https://github.com/networkx/networkx-metis/blob/master/nxmetis/enums.py
    METIS_opt = {
        'seed': getattr(args, "seed", None),
        'nseps': getattr(args, "nseps", None),
        'niter': getattr(args, "metis_niter", None),
        'compress': False,  # NOTE: this is differnt from default!
        'ncuts': getattr(args, "ncuts", None),
        # 0, edgecut, 1 Vol minimization! # NOTE: this is differnt from default edgecut.
        'objtype': getattr(args, 'objtype', None),
        # NOTE: default is -1, # TODO: add getattr getattr(args, "metis_dbglvl", None),
        '_dbglvl': 1  # TODO: can't make it print...
    }

    #     {'ptype': -1,
    #  'objtype': -1,
    #  'ctype': -1,
    #  'iptype': -1,
    #  'rtype': -1,
    #  'ncuts': -1,
    #  'nseps': -1,
    #  'numbering': -1,
    #  'niter': -1, # default is 10
    #  'seed': -1,
    #  'minconn': True,
    #  'no2hop': True,
    #  'contig': True,
    #  'compress': True,
    #  'ccorder': True,
    #  'pfactor': -1,
    #  'ufactor': -1,
    #  '_dbglvl': -1,
    #  }

    return args, METIS_opt


if __name__ == "__main__":
    args, METIS_opt = parse_cli()
    VERBOSE_PARTITIONING = False
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
    # DEBUG switches between verbose generated code and compressed code
    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    n_partitions = args.n_partitions
    graph = pipe_model(model,
                       sample,
                       depth=args.depth,
                       kwargs=None,
                       nparts=n_partitions,
                       DEBUG=VERBOSE_PARTITIONING,
                       output_file=args.output_file,
                       use_layers_only_graph=args.partition_layer_graph,
                       node_weight_function=node_weight_function,
                       edge_weight_function=edge_weight_function(bw),
                       n_iter=n_iter,
                       recomputation=recomputation,
                       save_memory_mode=args.save_memory_mode,
                       METIS_opt=METIS_opt)

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")
        graph.serialize(args.output_file)

    generated = importlib.import_module(args.output_file)
    create_pipeline_configuration = generated.create_pipeline_configuration

    if GET_PARTITIONS_ON_CPU:
        sample = sample.to('cpu')
    config = create_pipeline_configuration(model,
                                           partitions_only=False,
                                           DEBUG=GET_PARTITIONS_ON_CPU)

    # Test # TODO: can do it on GPU...
    if not args.no_test_run:
        _ = run_partitions(sample, config)

    if not args.no_analysis:
        sample = create_random_sample(args, analysis=True)
        analysis_result = run_analysis(sample,
                                       graph,
                                       config,
                                       n_iter,
                                       recomputation=recomputation,
                                       bw_GBps=bw,
                                       verbose=True,
                                       async_pipeline=args.async_pipeline)
