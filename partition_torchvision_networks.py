import os
from model_partition import partition_network_using_profiler, distribute_model
import torch
from sample_models import alexnet, resnet18, vgg11_bn, squeezenet1_0, inception_v3, densenet121, GoogLeNet, LeNet, WideResNet


def torchvision_write_traces():
    networks = [alexnet, resnet18, vgg11_bn, squeezenet1_0,
                inception_v3, densenet121, GoogLeNet, LeNet, WideResNet]
    for net in networks:
        model = net(pretrained=False).to("cuda:0")
        if net.__name__.find("inception") != -1:
            x = torch.zeros(10, 3, 299, 299)
        else:
            x = torch.zeros(10, 3, 224, 224)
        x = x.to("cuda:0")
        with torch.no_grad():
            trace_graph, _ = torch.jit.get_trace_graph(
                model, x)
            trace_graph = trace_graph.graph()
            trace = trace_graph.__str__()
            filename = f"{net.__name__}trace"
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = f"{curr_dir}\\net_traces"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(f"{out_dir}\\{filename}.txt", "w") as file:
                file.write(trace)


def partition_torchvision():
    networks = [alexnet, resnet18, vgg11_bn, squeezenet1_0,
                inception_v3, densenet121, GoogLeNet, LeNet, WideResNet]
    depth = [0, 1, 100]
    num_partitions = 4
    networks = [resnet18]
    depth = [100]
    for net in networks:
        model = net()
        for d in depth:
            print(f"current net is {net.__name__}")
            if net.__name__.find("inception") != -1:
                graph, _, _ = partition_network_using_profiler(
                    model, num_partitions, torch.zeros(4, 3, 299, 299), max_depth=d)
            elif net.__name__.find("GoogLeNet") != -1:
                graph, _, _ = partition_network_using_profiler(
                    model, num_partitions, torch.zeros(4, 3, 32, 32), max_depth=d)
            elif net.__name__.find("LeNet") != -1:
                graph, _, _ = partition_network_using_profiler(
                    model, num_partitions, torch.zeros(4, 3, 32, 32), max_depth=d)
            else:
                graph, _, _ = partition_network_using_profiler(
                    model, num_partitions, torch.zeros(4, 3, 224, 224), max_depth=d)

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
    networks = [resnet18]
    depth = [100]
    for net in networks:
        for d in depth:
            model = net()
            print(f"current net is {net.__name__}")
            if net.__name__.find("inception") != -1:
                _, graph, _ = distribute_model(model, torch.zeros(
                    4, 3, 299, 299), device_list=["cuda", "cpu"], num_iter=4, max_depth=d, basic_blocks=None)

            elif net.__name__.find("GoogLeNet") != -1:
                _, graph, _ = distribute_model(model, torch.zeros(
                    4, 3, 32, 32), device_list=["cuda", "cpu"], num_iter=4, max_depth=d, basic_blocks=None)

            elif net.__name__.find("LeNet") != -1:
                _, graph, _ = distribute_model(model, torch.zeros(
                    4, 3, 32, 32), device_list=["cuda", "cpu"], num_iter=4, max_depth=d, basic_blocks=None)

            else:
                _, graph, _ = distribute_model(model, torch.zeros(
                    4, 3, 224, 224), device_list=["cuda", "cpu"], num_iter=4, max_depth=d, basic_blocks=None)

            filename = f"{net.__name__} attempted {2} partitions at depth {d}"
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = f"{curr_dir}\\distribution_viz"
            graph.save(directory=out_dir, file_name=filename,
                       show_buffs_params=False, show_weights=False)

            print(filename)


if __name__ == "__main__":
    partition_torchvision()
