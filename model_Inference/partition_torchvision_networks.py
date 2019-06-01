import torch
import torch.nn as nn
import torchvision.models as models
from profile_and_partition import partition_network_using_profiler


def torchvision_write_traces():
    networks = [models.alexnet, models.resnet18, models.vgg11_bn,
                models.squeezenet1_0, models.inception_v3, models.densenet121]
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
            import os
            filename = f"{net.__name__}trace"
            directory = f"{os.getcwd()}\\traces"
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(f"{directory}\\{filename}.txt", "w") as file:
                file.write(trace)


def partition_torchvision():
    networks = [models.alexnet, models.resnet18, models.vgg11_bn,
                models.squeezenet1_0, models.inception_v3, models.densenet121]
    depth = [0, 1, 100]
    num_partitions = 4
    for net in networks:
        model = net()
        for d in depth:
            if net.__name__.find("inception") != -1:
                graph, _, _ = partition_network_using_profiler(
                    model, num_partitions, torch.zeros(10, 3, 299, 299), max_depth=d)
            else:
                graph, _, _ = partition_network_using_profiler(
                    model, num_partitions, torch.zeros(10, 3, 224, 224), max_depth=d)

            filename = f"{net.__name__} attempted {num_partitions} partitions at depth {d}"
            graph.save(directory="partitions", file_name=filename,
                       show_buffs_params=False, show_weights=False)
            print(filename)


if __name__ == "__main__":
    print("a")

    # partition_torchvision()
