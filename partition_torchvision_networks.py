import os
from pytorch_Gpipe import partition_with_profiler, distribute_using_profiler
import torch
from sample_models import alexnet, resnet18, vgg11_bn, squeezenet1_0, inception_v3, densenet121, GoogLeNet, LeNet, WideResNet


device = 'cpu' if not torch.cuda.is_available() else 'cuda'


def partition_torchvision():
    networks = [alexnet, resnet18, vgg11_bn, squeezenet1_0,
                inception_v3, densenet121, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    num_partitions = 4
    for net in networks:
        model = net().to(device)
        for d in depth:
            print(f"current net is {net.__name__}")
            if net.__name__.find("inception") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 299, 299).to(device), nparts=num_partitions, max_depth=d)
            elif net.__name__.find("GoogLeNet") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 32, 32).to(device), nparts=num_partitions, max_depth=d)
            elif net.__name__.find("LeNet") != -1:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 32, 32).to(device), nparts=num_partitions, max_depth=d)
            else:
                graph = partition_with_profiler(
                    model, torch.zeros(4, 3, 224, 224).to(device), nparts=num_partitions, max_depth=d)

            filename = f"{net.__name__} attempted {num_partitions} partitions at depth {d}"

            curr_dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = f"{curr_dir}\\partition_visualization"
            graph.save(directory=out_dir, file_name=filename,
                       show_buffs_params=False, show_weights=False)
            print(filename)


def distribute_torchvision():
    networks = [alexnet, resnet18, vgg11_bn, squeezenet1_0,
                inception_v3, densenet121, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    devices = ["cuda", "cpu", "cpu", "cuda"]
    for net in networks:
        for d in depth:
            model = net().to(device)
            print(f"current net is {net.__name__}")
            if net.__name__.find("inception") != -1:
                pipeline, graph, (counter, wrappers, sample_batch) = distribute_using_profiler(model, torch.zeros(
                    4, 3, 299, 299).to(device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

            elif net.__name__.find("GoogLeNet") != -1:
                pipeline, graph, _ = distribute_using_profiler(model, torch.zeros(
                    4, 3, 32, 32).to(device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

            elif net.__name__.find("LeNet") != -1:
                pipeline, graph, _ = distribute_using_profiler(model, torch.zeros(
                    4, 3, 32, 32).to(device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

            else:
                pipeline, graph, _ = distribute_using_profiler(model, torch.zeros(
                    4, 3, 224, 224).to(device), device_list=devices, num_iter=4, max_depth=d, basic_blocks=None)

            filename = f"{net.__name__} attempted {len(devices)} partitions at depth {d}"
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = f"{curr_dir}\\partition_visualization"
            graph.save(directory=out_dir, file_name=filename,
                       show_buffs_params=False, show_weights=False)

            print(filename)


if __name__ == "__main__":
    distribute_torchvision()
