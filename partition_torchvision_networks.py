import os
from pytorch_Gpipe import partition_with_profiler, profileNetwork, distribute_by_memory, distribute_by_time, \
    distribute_using_profiler, pipe_model
import torch
from sample_models import alexnet, resnet152, vgg19_bn, squeezenet1_1, inception_v3, densenet201, GoogLeNet, LeNet, \
    WideResNet
import torch.nn as nn
from pytorch_Gpipe.utils import model_scopes
from sample_models import AmoebaNet_D as my_amoeaba, amoebanetd as ref_amoeba, torchgpipe_resnet101
import datetime
from experiments.graph_serialization import serialize_graph


def partition_torchvision(networks=None, nparts=4, depth=100, nruns=4,
                          save_graph=False, show_scope_diff=False,
                          dump_graph=False, **model_kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if networks != None and not isinstance(networks, (list, tuple)):
        networks = [networks]

    if networks is None:
        networks = [alexnet, resnet152, torchgpipe_resnet101, vgg19_bn, squeezenet1_1,
                    inception_v3, densenet201, GoogLeNet, LeNet, WideResNet]

    if not isinstance(nparts, (list, tuple)):
        nparts = [nparts]

    if not isinstance(depth, (list, tuple)):
        depth = [depth]

    for i in range(nruns):
        for net in networks:
            model = net(**model_kwargs).to(device)
            print("model built")
            basic_blocks = None
            for p in nparts:
                for d in depth:
                    print(f"current net is {net.__name__}")
                    if net.__name__.find("inception") != -1:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 299, 299, device=device), nparts=p, max_depth=d,
                            basic_blocks=basic_blocks)
                    elif net.__name__.find("GoogLeNet") != -1:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device), nparts=p, max_depth=d,
                            basic_blocks=basic_blocks)
                    elif net.__name__.find("LeNet") != -1:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device), nparts=p, max_depth=d,
                            basic_blocks=basic_blocks)
                    else:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 224, 224, device=device), nparts=p, max_depth=d, basic_blocks=basic_blocks)

                    time_stemp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                    filename = f"{net.__name__}_run{i}_attempted_{p}_partitions_at_depth_{d}_{time_stemp}"

                    curr_dir = os.path.dirname(os.path.realpath(__file__))
                    out_dir = f"{curr_dir}\\partition_visualization"

                    if dump_graph:
                        serialize_graph(graph,
                                        f"{curr_dir}\\graph_dump\\{filename}")
                    if save_graph:
                        graph.save(directory=out_dir, file_name=filename,
                                   show_buffs_params=False, show_weights=False)
                    print(filename)

                    if show_scope_diff:
                        scopes = set(model_scopes(model, depth=d,
                                                  basic_blocks=basic_blocks))
                        graph_scopes = graph.scopes()
                        diff = scopes.difference(graph_scopes)
                        print(f"scope diff {len(diff)}")
                        for s in diff:
                            print(s)
                    print("\n")


def distribute_torchvision(networks=None, nparts=4, depth=100, nruns=4,
                           fake_gpus=False, save_graph=False, show_scope_diff=False,
                           optimize_pipeline_wrappers=True, dump_graph=False, **model_kwargs):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")

    device = 'cuda:0'
    if networks != None and not isinstance(networks, (list, tuple)):
        networks = [networks]

    if networks is None:
        networks = [my_amoeaba, ref_amoeba, alexnet, resnet152, torchgpipe_resnet101, vgg19_bn, squeezenet1_1,
                    inception_v3, densenet201, GoogLeNet, LeNet, WideResNet]

    if not isinstance(nparts, (list, tuple)):
        nparts = [nparts]

    if not isinstance(depth, (list, tuple)):
        depth = [depth]

    for i in range(nruns):
        for net in networks:
            model = net(**model_kwargs).to(device)
            print("model built")
            basic_blocks = None
            for p in nparts:
                if fake_gpus:
                    devices = [f'cuda:0' for _ in range(p)]
                else:
                    assert torch.cuda.device_count() == p
                    devices = [f'cuda:{i}' for i in range(p)]
                for d in depth:
                    print(f"current net is {net.__name__}")
                    if net.__name__.find("inception") != -1:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 299, 299, device=device),
                            optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d,
                            basic_blocks=basic_blocks)
                    elif net.__name__.find("GoogLeNet") != -1:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device),
                            optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d,
                            basic_blocks=basic_blocks)
                    elif net.__name__.find("LeNet") != -1:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device),
                            optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d,
                            basic_blocks=basic_blocks)
                    else:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 224, 224, device=device),
                            optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d,
                            basic_blocks=basic_blocks)

                    time_stemp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                    filename = f"{net.__name__}_run{i}_attempted_{p}_partitions_at_depth_{d}_{time_stemp}"

                    curr_dir = os.path.dirname(os.path.realpath(__file__))
                    out_dir = f"{curr_dir}\\distributed_models"
                    if dump_graph:
                        serialize_graph(graph,
                                        f"{curr_dir}\\graph_dump\\{filename}")
                    if save_graph:
                        graph.save(directory=out_dir, file_name=filename,
                                   show_buffs_params=False, show_weights=False)

                    if show_scope_diff:
                        scopes = set(model_scopes(model, depth=d,
                                                  basic_blocks=basic_blocks))
                        graph_scopes = graph.scopes()
                        diff = scopes.difference(graph_scopes)
                        print(f"scope diff {len(diff)}")
                        for s in diff:
                            print(s)
                    print("\n")


if __name__ == "__main__":
    partition_torchvision(
        networks=[my_amoeaba, ref_amoeba], nparts=8, save_graph=True, nruns=2, num_layers=5)
