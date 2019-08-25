from sample_models.amoebaNet import AmoebaNet_D
from experiments.graph_serialization import serialize_graph, deserialize_graph
import torch
from pytorch_Gpipe import distribute_using_profiler
from pytorch_Gpipe.pipeline.pipeline_parallel import PipelineParallel
from pytorch_Gpipe import distribute_by_time


def check_shape():
    num_parts = [2, 4]
    num_runs = 10

    net = AmoebaNet_D()
    model = net.to('cuda:0')
    x_tag = torch.randn((3, 3, 256, 256), device='cuda:0')
    y_tag = model(x_tag)

    for n_parts in num_parts:
        devices = [f'cuda:{i}' for i in range(n_parts)]

        for _ in range(num_runs):
            x = torch.randn((3, 3, 256, 256), device=devices[0])
            net = AmoebaNet_D()

            modified_model, wrappers, counter, _ = distribute_by_time(net, x,
                                                                      devices=devices, depth=100,
                                                                      optimize_pipeline_wrappers=True)
            in_shape = x.shape[1:]
            in_shape = tuple(in_shape)

            pipe = PipelineParallel(modified_model, 1, in_shape, wrappers, counter)

            y = pipe(x)

            if y.shape != y_tag.shape:
                print("different shape orig_shape:", y_tag.shape, " out_shape:", y.shape, " num_gpus = ", n_parts)


def check_backward():
    num_parts = [2, 4]
    num_runs = 10

    try:
        for n_parts in num_parts:
            devices = [f'cuda:{i}' for i in range(n_parts)]

            for _ in range(num_runs):
                x = torch.randn((3, 3, 256, 256), device=devices[0])
                net = AmoebaNet_D()

                modified_model, wrappers, counter, graph = distribute_by_time(net, x,
                                                                              devices=devices, depth=100,
                                                                              optimize_pipeline_wrappers=True)
                in_shape = x.shape[1:]
                in_shape = tuple(in_shape)

                pipe = PipelineParallel(modified_model, 1, in_shape, wrappers, counter)
                y = pipe(x)
                y.backward()
    except():
        serialize_graph(graph, "amoeba_net_backward.p")


if __name__ == '__main__':
    net = AmoebaNet_D()
    check_shape()
    check_backward()
