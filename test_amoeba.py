from sample_models.amoebaNet import AmoebaNet_D
from experiments.graph_serialization import serialize_graph, deserialize_graph
import torch
from pytorch_Gpipe import distribute_using_profiler
from pytorch_Gpipe.pipeline.pipeline_parallel import PipelineParallel
from pytorch_Gpipe import distribute_by_time


def check_shape(model_class, num_parts, num_runs=2, **kwargs):
    net = model_class(**kwargs)
    model = net.to('cuda:0')
    x_tag = torch.randn((3, 3, 224, 224), device='cuda:0')
    y_tag = model(x_tag)

    torch.cuda.synchronize()
    del model

    for n_parts in num_parts:
        devices = [f'cuda:{i}' for i in range(n_parts)]
        x = torch.randn((3, 3, 224, 224), device=devices[0])
        for i in range(num_runs):
            print(f"check shape {n_parts} run {i}")
            net = model_class(**kwargs).to(devices[0])

            modified_model, wrappers, counter, graph = distribute_by_time(net, x,
                                                                          devices=devices, depth=100,
                                                                          optimize_pipeline_wrappers=True)
            in_shape = x.shape[1:]
            in_shape = tuple(in_shape)

            pipe = PipelineParallel(
                modified_model, 1, in_shape, wrappers, counter)

            try:
                y = pipe(x)

                if y.shape != y_tag.shape:
                    print(f"check shape {n_parts} run {i} failed")
                    print("different shape orig_shape:", y_tag.shape,
                          " out_shape:", y.shape, " num_gpus = ", n_parts)
                else:
                    print(f"check shape {n_parts} run {i} good")

                torch.cuda.synchronize()
                del pipe
            except Exception as _:
                print("failed")
                serialize_graph(
                    graph, f"{model_class.__name__}_shape_{n_parts}_{i}.p")


def check_backward(model_class, num_parts, num_runs=2, **kwargs):
    for n_parts in num_parts:
        devices = [f'cuda:{i}' for i in range(n_parts)]
        x = torch.randn((3, 3, 224, 224), device=devices[0])
        for i in range(num_runs):
            print(f"check backward {n_parts} run {i}")
            try:
                net = model_class(**kwargs).to(devices[0])

                modified_model, wrappers, counter, graph = distribute_by_time(net, x,
                                                                              devices=devices, depth=100,
                                                                              optimize_pipeline_wrappers=True)
                in_shape = x.shape[1:]
                in_shape = tuple(in_shape)

                pipe = PipelineParallel(
                    modified_model, 1, in_shape, wrappers, counter)
                y = pipe(x)
                y.backward()
            except Exception as _:
                print("failed")
                serialize_graph(
                    graph, f"{model_class.__name__}_backward_{n_parts}_{i}.p")
            finally:
                torch.cuda.synchronize()
                del pipe


if __name__ == '__main__':
    print("visible devices")
    for device in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(device=device))

    num_parts = [2, 4]
    num_runs = 2
    print(
        f"begin sanity checks for {num_parts} partition options {num_runs} iterations")

    for p in num_parts:
        if p > torch.cuda.device_count():
            print(
                f"need {p} devices only {torch.cuda.device_count()} are visible")
            assert False

    print("\n now checking output shape\n")
    check_shape(AmoebaNet_D, num_parts, num_runs=num_runs, num_layers=1)
    print("\n now checking backward\n")
    check_backward(AmoebaNet_D, num_parts, num_runs=num_runs, num_layers=1)
