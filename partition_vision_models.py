import torch
from sample_models import WideResNet, AmoebaNet_D
from pytorch_Gpipe import pipe_model
import argparse
import importlib
from collections import deque
from misc import run_analysis, run_partitions

_WIDE_RESNETS = dict(
    wrn_16x4=dict(depth=16, num_classes=10, widen_factor=4,
                  drop_rate=0.0),  # FOR BACKWARD COMPATABILITY
    wrn_16x4_c10=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    wrn_16x4_c100=dict(depth=16, num_classes=100, widen_factor=4, drop_rate=0.0),
    wrn_28x10_c10_dr03=dict(depth=28, num_classes=10,widen_factor=10, drop_rate=0.3),
    wrn_28x10_c10=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0),
    wrn_28x10_c100_dr03=dict(depth=28, num_classes=100,
                             widen_factor=10, drop_rate=0.3),
    wrn_28x10_c100=dict(depth=28, num_classes=100,
                        widen_factor=10, drop_rate=0),
)

# this model is realy big even with 4 cells it contains 845 layers
_AMOEBANET_D = dict(
    amoebanet_4x512_c10=dict(num_layers=4, num_filters=512, num_classes=10),
    amoebanet_8x512_c100=dict(num_layers=8, num_filters=512, num_classes=100)
)

MODEL_CONFIGS = {**_WIDE_RESNETS, **_AMOEBANET_D}

# (note) originally used to get generated pipeline name later
MODEL_CFG_TO_SAMPLE_MODEL = {k: WideResNet for k in _WIDE_RESNETS.keys()}
MODEL_CFG_TO_SAMPLE_MODEL.update({k: AmoebaNet_D for k in _AMOEBANET_D.keys()})


DATASETS = ['cifar10', 'cifar100', 'imagenet']


def create_model(cfg='wrn_16x4'):
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg](**MODEL_CONFIGS[cfg])


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


def by_time(w):
    if hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
        return max(int(2 * (0 + w.backward_time) / 2), 1)
    return 0


def test_gpipe_stuff():

    # create a pipeLine from the given model
    # split dim the dim to split inputs and gradients across
    # DEBUG switches between running workers on CPU or GPUS

    partition_config = createConfig(model, partitions_only=False,
                                    DEBUG=GET_PARTITIONS_ON_CPU)
    output_device = 'cpu' if GET_PARTITIONS_ON_CPU else 'cuda'

    from pytorch_Gpipe import Pipeline
    pipe = Pipeline(partition_config, output_device=output_device,
                    split_dim=0, use_delayedNorm=False)

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

    parser = argparse.ArgumentParser(description="Partitioning models")
    parser.add_argument('--model', default='wrn_16x4',
                        choices=MODEL_CONFIGS.keys())
    parser.add_argument('--dataset', default='cifar10', choices=DATASETS)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_too_big', action='store_true', default=False,
                        help="if the model is too big run the whole partitioning process on CPU, and drink a cup of coffee in the meantime")
    parser.add_argument('--n_partitions', type=int, default=4)
    parser.add_argument('--output_file', default='wrn_16x4')
    parser.add_argument('--no_auto_file_name', action='store_true',
                        default=False, help="do not create file name automatically")
    parser.add_argument('--n_iter', type=int, default=100,
                        help="number of iteration used in order to profile the network and run analysis")
    parser.add_argument('--bandwidth_gps', type=float,
                        default=12, help="data transfer rate between gpus in gigabaytes per second")
    parser.add_argument('--no_recomputation', action='store_true',
                        default=False, help="wether to use recomputation for the backward pass")
    parser.add_argument('--no_analysis', action='store_true',
                        default=False, help="disable partition analysis")
    parser.add_argument("--depth", default=1000, type=int,
                        help="the depth in which we will partition the model")
    parser.add_argument("--analysis_batch_size", default=8, type=int,
                        help="batch size to use during the post partition analysis")

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

    if isinstance(model, AmoebaNet_D) and args.dataset != 'imagenet':
        error = "amoebanet supported input size is 3x224x244, use imagenet dataset instead"
        raise ValueError(error)

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
    graph = pipe_model(model, sample, depth=args.depth, kwargs=None, nparts=args.n_partitions,
                       DEBUG=VERBOSE_PARTITIONING, output_file=args.output_file, weight_func=by_time, n_iter=n_iter)
    graph.save(args.output_file, ".")

    generated = importlib.import_module(args.output_file)
    createConfig = generated.createConfig

    if GET_PARTITIONS_ON_CPU:
        sample = sample.to('cpu')
    config = createConfig(
        model, partitions_only=False, DEBUG=GET_PARTITIONS_ON_CPU)

    _ = run_partitions(sample, config)
    bandwidth_gps = args.bandwidth_gps
    recomputation = not args.no_recomputation
    if not args.no_analysis:
        sample = create_random_sample(args, analysis=True)
        run_analysis(sample, graph, config, n_iter,
                     recomputation=recomputation, bandwidth_gps=bandwidth_gps)
    # test_gpipe_stuff()
