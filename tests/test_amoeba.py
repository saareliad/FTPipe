from sample_models.amoebaNet import AmoebaNet_D
from experiments.graph_serialization import serialize_graph, deserialize_graph
import torch
from pytorch_Gpipe import distribute_using_profiler


def check_shape(net):
    num_parts = [2, 4]
    num_runs = 10

    model = net.to('cuda:0')
    x_tag = torch.randn((3, 3, 256, 256), device='cuda:0')
    y_tag = model(x_tag)

    for n_parts in num_parts:
        devices = [f'cuda:{i}' for i in range(n_parts)]

        for _ in range(num_runs):
            x = torch.randn((3, 3, 256, 256), device=devices[0])

            model, _, _, graph = distribute_using_profiler(
                net, torch.zeros(3, 3, 256, 256, device=devices[0]),
                optimize_pipeline_wrappers=True, devices=devices, max_depth=100,
                basic_blocks=None)

            y = model(x)

            if y.shape != y_tag.shape:
                print("different shape orig_shape:", y_tag.shape, " out_shape:", y.shape, " num_gpus = ", n_parts)


def check_backward(net):
    num_parts = [2, 4]
    num_runs = 10

    try:
        for n_parts in num_parts:
            devices = [f'cuda:{i}' for i in range(n_parts)]

            for _ in range(num_runs):
                x = torch.randn((3, 3, 256, 256))

                model, _, _, graph = distribute_using_profiler(
                    net, torch.zeros(3, 3, 256, 256, device=devices[0]),
                    optimize_pipeline_wrappers=True, devices=devices, max_depth=100,
                    basic_blocks=None)

                y = model(x)
                y.backward()
    except():
        serialize_graph(graph, "amoeba_net_backward.p")


if __name__ == '__main__':
    net = AmoebaNet_D()
    check_shape(net)
    check_backward(net)
