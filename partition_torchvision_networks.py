from collections import OrderedDict
import os
from pytorch_Gpipe import partition_with_profiler, profileNetwork, distribute_by_memory, distribute_by_time, distribute_using_profiler, pipe_model
import torch
from sample_models import alexnet, resnet152, vgg19_bn, squeezenet1_1, inception_v3, densenet201, GoogLeNet, LeNet, WideResNet
import torch.nn as nn
from pytorch_Gpipe.utils import model_scopes
from sample_models import AmoebaNet_D as my_amoeaba, amoebanetd as ref_amoeba, torchgpipe_resnet101


def partition_torchvision(nparts=4, save_graph=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    networks = [my_amoeaba, ref_amoeba, alexnet, resnet152, torchgpipe_resnet101, vgg19_bn, squeezenet1_1,
                inception_v3, densenet201, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    for net in networks:
        model = net().to(device)
        for d in depth:
            print(f"current net is {net.__name__}")
            if net.__name__.find("inception") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(16, 3, 299, 299, device=device), nparts=nparts, max_depth=d)
            elif net.__name__.find("GoogLeNet") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(16, 3, 32, 32, device=device), nparts=nparts, max_depth=d)
            elif net.__name__.find("LeNet") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(16, 3, 32, 32, device=device), nparts=nparts, max_depth=d)
            else:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 224, 224, device=device), nparts=nparts, max_depth=d)

            filename = f"{net.__name__} attempted {nparts} partitions at depth {d}"

            curr_dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = f"{curr_dir}\\partition_visualization"
            if save_graph:
                graph.save(directory=out_dir, file_name=filename,
                           show_buffs_params=False, show_weights=False)
            print(filename)


def distribute_torchvision(nruns=1, nparts=4, save_graph=False):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")

    device = 'cuda'
    devices = ['cuda' for _ in range(nparts)]
    networks = [alexnet, resnet152, vgg19_bn, squeezenet1_1,
                inception_v3, densenet201, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    for idx in range(nruns):
        for net in networks:
            for d in depth:
                print()
                filename = f"{net.__name__} {nparts} partitions at depth {d} attempt {idx}"
                curr_dir = os.path.dirname(os.path.realpath(__file__))
                out_dir = f"{curr_dir}\\graphs"

                print(filename)
                model = net().to(device)
                if net.__name__.find("inception") != -1:
                    _, _, _, graph = distribute_using_profiler(model, torch.zeros(
                        4, 3, 299, 299, device=device), devices=devices, max_depth=d, basic_blocks=None)

                elif net.__name__.find("GoogLeNet") != -1:
                    _, _, _, graph = distribute_using_profiler(model, torch.zeros(
                        4, 3, 32, 32, device=device), devices=devices, max_depth=d, basic_blocks=None)

                elif net.__name__.find("LeNet") != -1:
                    _, _, _, graph = distribute_using_profiler(model, torch.zeros(
                        4, 3, 32, 32, device=device), devices=devices, max_depth=d, basic_blocks=None)

                else:
                    _, _, _, graph = distribute_using_profiler(model, torch.zeros(
                        4, 3, 224, 224, device=device), devices=devices, max_depth=d, basic_blocks=None)
                if save_graph:
                    graph.save(directory=out_dir, file_name=filename,
                               show_buffs_params=False, show_weights=False)

                print(filename)
                print()


def compare_exec_time():
    device = torch.device('cuda:0')
    net = densenet201().to(device)
    torch.cuda.synchronize()
    x = torch.randn(16, 3, 224, 224, device=device)
    x = net(x)
    torch.cuda.synchronize()
    for _ in range(2):
        # milliseconds
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        x = torch.randn(16, 3, 224, 224, device=device)
        x = net(x)
        end.record()
        torch.cuda.synchronize()
        f_time = (start.elapsed_time(end))
        print(f"f_time {f_time}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        x = torch.randn(16, 3, 224, 224, device=device)
        y = net(x)
        torch.cuda.synchronize()
        start.record()
        loss = y.norm()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        b_time = (start.elapsed_time(end))
        print(f"b_time {b_time}")

    for _ in range(2):
        x = torch.randn(16, 3, 224, 224, device=device)
        profiles = profileNetwork(net, x, max_depth=0)
        torch.cuda.synchronize(device)

        profs = profiles.values()
        profs = list(filter(lambda p: hasattr(p, 'forward_time')
                            and hasattr(p, 'backward_time'), profs))

        profs = list(map(lambda p: (p.forward_time, p.backward_time), profs))
        print(profs)


def compare_cuda_mem():
    device = torch.device('cuda:0')
    num_init_features = 64

    def init():
        net = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)), ])).to(device)
        tensor = torch.randn(4, 3, 224, 224, device=device)
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_max_memory_allocated(device=device)
        return net, tensor

    features, x = init()
    names = ['conv0', 'norm0', 'relu0', 'pool0']
    expected_mem = dict()
    for l, n in zip(features, names):
        torch.cuda.reset_max_memory_allocated(device=device)
        torch.cuda.synchronize(device)
        x = l(x)
        torch.cuda.synchronize(device)
        expected = torch.cuda.max_memory_allocated(device=device)/1e9
        expected_mem[n] = expected

    features, x = init()

    profiles = profileNetwork(features, x)
    diffs = dict()
    for n, p in profiles.items():
        actual = p.cuda_memory_forward
        for other in names:
            if other in n:
                expected = expected_mem[other]
                diffs[n] = abs(expected-actual)

    print(diffs)


def integration():
    device = 'cuda'
    devices = ['cuda:0', 'cuda:1']
    x = torch.randn(16, 3, 224, 224, device=device)
    net = alexnet().to(device)
    pipe_net = pipe_model(net,  8, x, devices=devices)
    pipe_net(x)


def tuple_problem():

    class dummy(nn.Module):
        def __init__(self, first=True):
            super(dummy, self).__init__()
            self.layer = nn.Linear(10, 10)
            self.first = first

        def forward(self, *xs):
            if isinstance(xs[0], tuple):
                return self.t_forward(xs[0])
            return self.m_forward(*xs)

        def t_forward(self, xs):
            x0, x1 = xs
            return self.m_forward(x0, x1)

        def m_forward(self, x0, x1):
            if self.first:
                return self.layer(x0), x1
            return x0, self.layer(x1)

    class seqDummy(nn.Module):
        def __init__(self, tupled):
            super(seqDummy, self).__init__()
            self.tupled = tupled

            self.t0 = dummy()
            self.t1 = dummy(first=False)

        def forward(self, *xs):
            if self.tupled:
                return self.t_forward(xs[0])
            return self.m_forward(*xs)

        def t_forward(self, xs):
            x0, x1 = xs
            return self.m_forward(x0, x1)

        def m_forward(self, x0, x1):
            a, b = self.t0(x0, x1)
            return self.t1(a, b)

    tupled = seqDummy(True)
    multi = seqDummy(False)

    g1 = partition_with_profiler(
        tupled, (torch.zeros(4, 10), torch.zeros(10, 10)), nparts=2)

    g2 = partition_with_profiler(
        multi, torch.zeros(4, 10), torch.zeros(10, 10), nparts=2)

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = f"{curr_dir}\\partition_visualization"

    g1.save(directory=out_dir, file_name="tupled",
            show_buffs_params=False, show_weights=False)
    g2.save(directory=out_dir, file_name="multi",
            show_buffs_params=False, show_weights=False)


if __name__ == "__main__":
    # integration()
    # partition_torchvision(nparts=2)
    # compare_exec_time()

    tuple_problem()
