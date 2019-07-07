from collections import OrderedDict
import os
from pytorch_Gpipe import partition_with_profiler, distribute_using_profiler
import torch
from sample_models import alexnet, resnet18, vgg11_bn, squeezenet1_0, inception_v3, densenet121, GoogLeNet, LeNet, WideResNet
import torch.nn as nn


def partition_torchvision(nparts=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    networks = [alexnet, resnet18, vgg11_bn, squeezenet1_0,
                inception_v3, densenet121, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    for net in networks:
        model = net().to(device)
        for d in depth:
            print(f"current net is {net.__name__}")
            if net.__name__.find("inception") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 299, 299, device=device), nparts=nparts, max_depth=d)
            elif net.__name__.find("GoogLeNet") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 32, 32, device=device), nparts=nparts, max_depth=d)
            elif net.__name__.find("LeNet") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 32, 32, device=device), nparts=nparts, max_depth=d)
            else:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 224, 224, device=device), nparts=nparts, max_depth=d)

            filename = f"{net.__name__} attempted {nparts} partitions at depth {d}"

            curr_dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = f"{curr_dir}\\partition_visualization"
            graph.save(directory=out_dir, file_name=filename,
                       show_buffs_params=False, show_weights=False)
            print(filename)


def distribute_torchvision(nruns=1, nparts=4):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")

    device = 'cuda'
    networks = [alexnet, resnet18, vgg11_bn, squeezenet1_0,
                inception_v3, densenet121, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    devices = ['cuda' for _ in range(nparts)]
    depth = [0]
    networks = [densenet121]
    for idx in range(nruns):
        for net in networks:
            for d in depth:
                model = net().to(device)
                print(f"current net is {net.__name__}")
                if net.__name__.find("inception") != -1:
                    pipeline, graph, (counter, wrappers, sample_batch) = distribute_using_profiler(model, torch.zeros(
                        4, 3, 299, 299, device=device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

                elif net.__name__.find("GoogLeNet") != -1:
                    pipeline, graph, _ = distribute_using_profiler(model, torch.zeros(
                        4, 3, 32, 32, device=device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

                elif net.__name__.find("LeNet") != -1:
                    pipeline, graph, _ = distribute_using_profiler(model, torch.zeros(
                        4, 3, 32, 32, device=device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

                else:
                    pipeline, graph, _ = distribute_using_profiler(model, torch.zeros(
                        4, 3, 224, 224, device=device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

                filename = f"{net.__name__} {nparts} partitions at depth {d} attempt {idx}"
                curr_dir = os.path.dirname(os.path.realpath(__file__))
                out_dir = f"{curr_dir}\\graphs"
                # graph.save(directory=out_dir, file_name=filename,
                #            show_buffs_params=False, show_weights=False)

                print(filename)


def test_alloc_time(*dims, immediate=False):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if immediate:
        start.record()
        a = torch.randn(*dims, device="cuda:0")
        end.record()
    else:
        start.record()
        a = torch.randn(*dims).to('cuda:0')
        end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))


def test_exec_time():
    num_init_features = 64
    features = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(3, num_init_features,
                            kernel_size=7, stride=2, padding=3, bias=False)),
        ('norm0', nn.BatchNorm2d(num_init_features)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ])).to('cuda:0')

    for _ in range(5):
        # milliseconds
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        x = torch.randn(4, 3, 224, 224, device='cuda:0')
        x = features(x)
        end.record()
        torch.cuda.synchronize()
        f_time = (start.elapsed_time(end))
        print(f_time)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        x = torch.randn(4, 3, 224, 224, device='cuda:0')
        y = features(x)
        loss = y.norm()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        b_time = (start.elapsed_time(end))
        print(b_time)


if __name__ == "__main__":
    test_exec_time()
