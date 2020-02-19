import torch
from sample_models import WideResNet, amoebanetd
from sample_models.ResNet import ResNet, Bottleneck
from pytorch_Gpipe import pipe_model
from pytorch_Gpipe.model_profiling import Node, NodeTypes
import argparse
import importlib
from misc import run_analysis, run_partitions

_RESENETS = dict(resnet50_imagenet=dict(
    block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000))

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
_register_model(_RESENETS, ResNet)
_register_model(_AMOEBANET_D, amoebanetd)

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


def node_weight_function(node: Node):
    if node.type is NodeTypes.LAYER:
        return int(node.weight.forward_time + node.weight.backward_time)
    if node.type is NodeTypes.CONSTANT:
        return 0
    if node.type is NodeTypes.OP:
        return 1
    return 0


def edge_weight_function(bandwidth_gps):
    def f(u: Node, v: Node):
        if u.type is NodeTypes.LAYER:
            return max(1, int(u.weight.output_size / bandwidth_gps))
            # print(f"u.weight.output_size:{u.weight.output_size}")  # FIXME: this fucntion is never called...
        if v.type is NodeTypes.LAYER:
            return max(1, int(v.weight.input_size / bandwidth_gps))
            # print(f"u.weight.output_size:{v.weight.input_size}")  # FIXME: this fucntion is never called...
        if u.type is NodeTypes.CONSTANT:
            return 1000
        return 1

    return f


def test_gpipe_stuff():

    # create a pipeLine from the given model
    # split dim the dim to split inputs and gradients across
    # DEBUG switches between running workers on CPU or GPUS

    partition_config = create_pipeline_configuration(
        model, partitions_only=False, DEBUG=GET_PARTITIONS_ON_CPU)
    output_device = 'cpu' if GET_PARTITIONS_ON_CPU else 'cuda'

    from pytorch_Gpipe import Pipeline
    pipe = Pipeline(partition_config,
                    output_device=output_device,
                    split_dim=0,
                    use_delayedNorm=False)

    output = pipe(sample.cpu())

    # compute loss
    loss0 = output.sum()
    loss1 = output.abs().sum()
    losses = [loss0, loss1]

    # compute gradients of the losses in respect to model outputs
    grads = torch.autograd.grad(losses, [output])

    # pass gradients to the pipeline and compute the backward pass
    pipe.backward(grads)


if __name__ == "__main__":

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
        help=
        "if the model is too big run the whole partitioning process on CPU, "
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
        help=
        "number of iteration used in order to profile the network and run analysis"
    )
    parser.add_argument(
        '--bandwidth_gps',
        type=float,
        default=12,
        help="data transfer rate between gpus in gigabaytes per second")
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
                        default=1000,
                        type=int,
                        help="the depth in which we will partition the model")
    parser.add_argument(
        "--partition_layer_graph",
        action="store_true",
        default=False,
        help="whether to partition a graph containing only layers")
    parser.add_argument(
        "--analysis_batch_size",
        default=8,
        type=int,
        help="batch size to use during the post partition analysis")
    parser.add_argument("-a", "--async_pipeline",
                        default=False,
                        action="store_true",
                        help="Do analysis for async pipeline")
    parser.add_argument("--dot", default=False, action="store_true", help="Save and plot it using graphviz")
    args = parser.parse_args()
    args.auto_file_name = not args.no_auto_file_name
    if args.auto_file_name:
        args.output_file = f"{args.model}_p{args.n_partitions}"

    VERBOSE_PARTITIONING = False
    GET_PARTITIONS_ON_CPU = True

    # if the model is too big run the whole partitioning process on CPU
    # and drink a cup of coffee in the meantime
    # define model and sample batch
    model = create_model(args.model)
    sample = create_random_sample(args, analysis=False)

    if args.model_too_big:
        model = model.cpu()
        sample = sample.cpu()
    else:
        model = model.cuda()
        sample = sample.cuda()

    # partition the model using our profiler
    # if the model need multiple inputs pass a tuple
    # if the model needs kwargs pass a dictionary
    # DEBUG switches between verbose generated code and compressed code
    n_iter = args.n_iter
    graph = pipe_model(model,
                       sample,
                       depth=args.depth,
                       kwargs=None,
                       nparts=args.n_partitions,
                       DEBUG=VERBOSE_PARTITIONING,
                       output_file=args.output_file,
                       use_layers_only_graph=args.partition_layer_graph,
                       node_weight_function=node_weight_function,
                       edge_weight_function=edge_weight_function(
                           args.bandwidth_gps),
                       n_iter=n_iter)

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

    _ = run_partitions(sample, config)
    bandwidth_gps = args.bandwidth_gps
    recomputation = not args.no_recomputation
    if not args.no_analysis:
        sample = create_random_sample(args, analysis=True)
        analysis_result = run_analysis(sample,
                                       graph,
                                       config,
                                       n_iter,
                                       recomputation=recomputation,
                                       bandwidth_gps=bandwidth_gps,
                                       verbose=True,
                                       async_pipeline=args.async_pipeline)
    # test_gpipe_stuff()
