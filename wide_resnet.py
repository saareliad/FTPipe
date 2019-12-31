import torch
from sample_models import WideResNet
from pytorch_Gpipe import pipe_model
import argparse
import importlib
import time

_WIDE_RESNETS = dict(
    wrn_16x4=dict(depth=16, num_classes=10, widen_factor=4,
                  drop_rate=0.0),  # FOR BACKWARD COMPATABILITY
    wrn_16x4_c10=dict(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0),
    wrn_28x10_c10_dr03=dict(depth=28, num_classes=10,
                            widen_factor=10, drop_rate=0.3),
    wrn_28x10_c10=dict(depth=28, num_classes=10, widen_factor=10, drop_rate=0),

    wrn_16x4_c100=dict(depth=16, num_classes=100,
                       widen_factor=4, drop_rate=0.0),
    wrn_28x10_c100_dr03=dict(depth=28, num_classes=100,
                             widen_factor=10, drop_rate=0.3),
    wrn_28x10_c100=dict(depth=28, num_classes=100,
                        widen_factor=10, drop_rate=0),
)

MODEL_CONFIGS = {**_WIDE_RESNETS}

# (note) originally used to get generated pipeline name later
MODEL_CFG_TO_SAMPLE_MODEL = {k: WideResNet for k in _WIDE_RESNETS.keys()}


DATASETS = ['cifar10', 'cifar100', 'imagenet']


def create_model(cfg='wrn_16x4'):
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg](**MODEL_CONFIGS[cfg])


def create_random_sample(args):
    dataset = args.dataset
    batch_size = args.batch_size

    if dataset == 'cifar10' or dataset == 'cifar100':
        sample = torch.randn(batch_size, 3, 32, 32)
    elif dataset == 'imagenet':
        sample = torch.randn(batch_size, 3, 224, 224)

    return sample


def by_time(w):
    if hasattr(w, 'forward_time') and hasattr(w, 'backward_time'):
        return max(int(200 * (0 + w.backward_time) / 2), 1)
    return 0


def extract_time(w, forward=False):
    if not hasattr(w, "forward_time"):
        return 0
    if forward:
        return w.forward_time
    return w.backward_time


def cuda_time(model, inputs, forward=False):
    model = model.cuda()
    inputs = [i.detach().cuda() for i in inputs]
    torch.cuda.synchronize(device='cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model(*inputs)
    end.record()
    torch.cuda.synchronize(device='cuda')
    exec_time = (start.elapsed_time(end))

    if forward:
        return exec_time, out

    loss = sum(o.norm() for o in out)
    torch.cuda.synchronize(device='cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize(device='cuda')
    exec_time = (start.elapsed_time(end))

    return exec_time, out


def cpu_time(model, inputs, forward=False):
    model = model.cpu()
    inputs = [i.detach().cpu() for i in inputs]
    start = time.time()
    out = model(*inputs)
    end = time.time()
    exec_time = 1000 * (end - start)

    if forward:
        return exec_time, out

    loss = sum(o.norm() for o in out)
    start = time.time()
    loss.backward()
    end = time.time()
    exec_time = 1000 * (end - start)

    return exec_time, out


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
    parser.add_argument('--auto_file_name', action='store_true',
                        default=False, help="create file name automatically")

    args = parser.parse_args()

    if args.auto_file_name:
        args.output_file = f"{args.model}_p{args.n_partitions}"

    VERBOSE_PARTITIONING = False
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
    n_iter = 10
    graph = pipe_model(model, sample, kwargs=None, nparts=args.n_partitions,
                       DEBUG=VERBOSE_PARTITIONING, output_file=args.output_file, weight_func=by_time, n_iter=n_iter)
    graph.save(args.output_file, ".")

    generated = importlib.import_module(args.output_file)
    createConfig = generated.createConfig

    if GET_PARTITIONS_ON_CPU:
        sample = sample.to('cpu')
    partitions = createConfig(
        model, partitions_only=True, DEBUG=GET_PARTITIONS_ON_CPU)

    def run_sequential(sample, partitions):
        if not isinstance(sample, tuple):
            sample = (sample,)
        out = sample
        for p in partitions:
            out = p(*out)
        return out

    out = run_sequential(sample, partitions)

    def test_gpipe_stuff():

        # create a pipeLine from the given model
        # split dim the dim to split inputs and gradients across
        # DEBUG switches between running workers on CPU or GPUS

        config = createConfig(model, partitions_only=False,
                              DEBUG=GET_PARTITIONS_ON_CPU)
        output_device = 'cpu' if GET_PARTITIONS_ON_CPU else 'cuda'

        from pytorch_Gpipe import Pipeline
        pipe = Pipeline(config, output_device=output_device,
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

    # TODO assumes sequential partition with no skip connections between parts aka i->i+1... and not i->i+2
    # can generalize
    def actual_imbalance(sample, partitions, forward=False):
        times = {i: 0 for i in range(args.n_partitions)}

        communication_volume = {}

        if not isinstance(sample, tuple):
            sample = (sample,)

        for _ in range(n_iter):
            out = sample
            for idx, p in enumerate(partitions):
                in_size = 0
                for i in out:
                    in_size += (i.nelement() * i.element_size()) / 1e6
                inOut = f"in: {in_size} MB "

                if torch.cuda.is_available():
                    exec_time, out = cuda_time(p, out, forward=forward)
                else:
                    exec_time, out = cpu_time(p, out, forward=forward)

                out_size = 0
                for o in out:
                    out_size += (o.nelement() * o.element_size()) / 1e6

                communication_volume[idx] = f"{inOut} out: {out_size} MB"
                times[idx] += exec_time

        avg_times = {i: v/n_iter for i, v in times.items()}

        return avg_times, communication_volume

    cutting_edges = 0
    theoretical_times = {i: 0 for i in range(args.n_partitions)}
    for n in graph.nodes:
        theoretical_times[n.part] += extract_time(n.weight,
                                                  forward=False)
        for u in n.out_nodes:
            if n.part != u.part:
                cutting_edges += 1
    print(f"number of cutting edges: {cutting_edges}")

    actual_times, comm_volume = actual_imbalance(sample,
                                                 partitions, forward=False)
    theoretical_imbalance = min(
        theoretical_times.values()) / max(theoretical_times.values())

    real_imbalance = min(actual_times.values())/max(actual_times.values())
    print(f"theoretical imbalance is: {theoretical_imbalance}")
    print(f"real imbalance is: {real_imbalance}")
    print(f"theoretical times ms {theoretical_times}")
    print(f"real times ms {actual_times}")
    print(
        f"communication volumes size of activations of each partition\n{comm_volume}")
    # test_gpipe_stuff()
