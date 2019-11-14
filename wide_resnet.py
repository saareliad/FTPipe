import torch
from sample_models import WideResNet
from pytorch_Gpipe import pipe_model
import argparse
import importlib


MODEL_CONFIGS = dict(wrn_16x4=dict(
    depth=16, num_classes=10, widen_factor=4, drop_rate=0.0))
DATASETS = ['cifar10', 'cifar100', 'imagenet']

# Used to get generated name later
MODEL_CFG_TO_SAMPLE_MODEL = dict(wrn_16x4=WideResNet)


def create_model(cfg='wrn_16x4'):
    return WideResNet(**MODEL_CONFIGS[cfg])


def create_random_sample(args):
    dataset = args.dataset
    batch_size = args.batch_size

    if dataset == 'cifar10' or dataset == 'cifar100':
        sample = torch.randn(batch_size, 3, 32, 32)
    elif dataset == 'imagenet':
        sample = torch.randn(batch_size, 3, 224, 224)

    return sample


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Partitioning models")
    parser.add_argument('--model', default='wrn_16x4',
                        choices=MODEL_CONFIGS.keys())
    parser.add_argument('--dataset', default='cifar10', choices=DATASETS)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_too_big', default=False,
                        help="if the model is too big run the whole partitioning process on CPU, and drink a cup of coffee in the meantime")
    parser.add_argument('--n_partitions', type=int, default=4)
    parser.add_argument('--output_file', default='wrn_16x4')

    args = parser.parse_args()
    VERBOSE_PARTITIONING = True
    GET_PARTITIONS_ON_CPU = True

    # if the model is too big run the whole partitioning process on CPU
    # and drink a cup of coffee in the meantime
    # define model and sample batch
    model = create_model(args.model)
    sample = create_random_sample(args)

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
    graph = pipe_model(model, sample, kwargs=None, nparts=args.n_partitions,
                       DEBUG=VERBOSE_PARTITIONING, output_file=args.output_file)

    # from PartitionedWideResnet101_2 import ResNetPipeline, createConfig
    # after that import the generated pipeline from the output file
    # It is called ResNetPipeline because the model class is named ResNet

    generated = importlib.import_module(args.output_file)
    createConfig = generated.createConfig

    if GET_PARTITIONS_ON_CPU:
        sample = sample.to('cpu')
    partitions = createConfig(
        model, partitions_only=True, DEBUG=GET_PARTITIONS_ON_CPU)

    def run_sequential(sample, partitions):
        a = partitions[0](sample)
        for i in range(1, args.n_partitions - 1):
            a = partitions[i](*a)
        out = partitions[args.n_partitions - 1](*a)
        return out

    out = run_sequential(sample, partitions)

    def test_gpipe_stuff():
        # FIXME: NOT TESTED YET. DECOUPLED FROM PARTITIONS.
        name = MODEL_CFG_TO_SAMPLE_MODEL[args.model].__name__
        gpipe_generated_pipeline = importlib.import_module(
            args.output_file + "." + name + "Pipeline")

        # create a pipeLine from the given model
        # split dim the dim to split inputs and gradients across
        # DEBUG switches between running workers on CPU or GPUS
        pipe = gpipe_generated_pipeline(
            model, output_device='cpu', split_dim=0, DEBUG=True)

        output = pipe(sample.cpu())

        # compute loss
        loss0 = output.sum()
        loss1 = output.abs().sum()
        losses = [loss0, loss1]

        # compute gradients of the losses in respect to model outputs
        grads = torch.autograd.grad(losses, [output])

        # pass gradients to the pipeline and compute the backward pass
        pipe.backward(grads)
