import torch
import torch.nn as nn
import torchvision.models as models
from network_profiler import profileNetwork
from control_flow_graph import build_control_flow_graph
from METIS_graph_partition import part_graph
from res_net_example import resnet20_cifar


def partition_model(model, num_gpus, *sample_batch, num_iter=4, max_depth=100, basic_blocks=None, device="cuda", weights=None, wrappers=None):

    if weights is None:
        weights = profileNetwork(model, *sample_batch, max_depth=max_depth,
                                 basic_block=basic_blocks, device=device, num_iter=num_iter)

    graph = build_control_flow_graph(
        model, *sample_batch, max_depth=max_depth, weights=weights, basic_block=basic_blocks, device=device)

    adjlist = graph.adjacency_list()
    nodew = graph.get_weights()

    assert(len(adjlist) == len(nodew))

    weights = [weight_func(w) for w in nodew]

    cuts, parts = part_graph(
        adjlist, nparts=num_gpus, algorithm="metis", nodew=weights, contig=1)

    graph.set_partition(parts)
    return graph, cuts, parts  # partition_cost(weights, parts, cuts)


# TODO decide on weighting functional
def weight_func(w):
    if isinstance(w, tuple):
        return int(100*(w.forward_time+w.backward_time)/2)
    return 0


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
            try:
                if net.__name__.find("inception") != -1:
                    graph, _, _ = partition_model(
                        model, num_partitions, torch.zeros(10, 3, 299, 299), max_depth=d)
                else:
                    graph, _, _ = partition_model(
                        model, num_partitions, torch.zeros(10, 3, 224, 224), max_depth=d)

                filename = f"{net.__name__} attempted {num_partitions} partitions at depth {d}"
                graph.save(directory="partitions", file_name=filename,
                           show_buffs_params=False, show_weights=False)
                print(filename)

            except Exception as e:
                print(e)


if __name__ == "__main__":
    partition_torchvision()
