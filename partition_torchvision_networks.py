import os
from pytorch_Gpipe import partition_with_profiler, profileNetwork, distribute_by_memory, distribute_by_time, distribute_using_profiler, pipe_model
import torch
from sample_models import alexnet, resnet152, vgg19_bn, squeezenet1_1, inception_v3, densenet201, GoogLeNet, LeNet, WideResNet
import torch.nn as nn
from pytorch_Gpipe.utils import model_scopes
from sample_models import AmoebaNet_D as my_amoeaba, amoebanetd as ref_amoeba, torchgpipe_resnet101


def partition_torchvision(networks=None, nparts=4, depth=100, nruns=4, save_graph=False, show_scope_diff=False):
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
            model = net().to(device)
            print("model built")
            basic_blocks = None
            for p in nparts:
                for d in depth:
                    print(f"current net is {net.__name__}")
                    if net.__name__.find("inception") != -1:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 299, 299, device=device), nparts=p, max_depth=d, basic_blocks=basic_blocks)
                    elif net.__name__.find("GoogLeNet") != -1:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device), nparts=p, max_depth=d, basic_blocks=basic_blocks)
                    elif net.__name__.find("LeNet") != -1:
                        graph = partition_with_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device), nparts=p, max_depth=d, basic_blocks=basic_blocks)
                    else:
                        graph = partition_with_profiler(
                            model, torch.zeros(4, 3, 224, 224, device=device), nparts=p, max_depth=d, basic_blocks=basic_blocks)

                    filename = f"{net.__name__}_run{i}_attempted_{p}_partitions_at_depth_{d}"

                    curr_dir = os.path.dirname(os.path.realpath(__file__))
                    out_dir = f"{curr_dir}\\partition_visualization"
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


def distribute_torchvision(networks=None, nparts=4, depth=100, nruns=4, fake_gpus=False, save_graph=False, show_scope_diff=False, optimize_pipeline_wrappers=True):
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
            model = net().to(device)
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
                            model, torch.zeros(16, 3, 299, 299, device=device), optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d, basic_blocks=basic_blocks)
                    elif net.__name__.find("GoogLeNet") != -1:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device), optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d, basic_blocks=basic_blocks)
                    elif net.__name__.find("LeNet") != -1:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 32, 32, device=device), optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d, basic_blocks=basic_blocks)
                    else:
                        model, _, _, graph = distribute_using_profiler(
                            model, torch.zeros(16, 3, 224, 224, device=device), optimize_pipeline_wrappers=optimize_pipeline_wrappers, devices=devices, max_depth=d, basic_blocks=basic_blocks)

                    filename = f"{net.__name__}_run{i}_attempted_{p}_partitions_at_depth_{d}"

                    curr_dir = os.path.dirname(os.path.realpath(__file__))
                    out_dir = f"{curr_dir}\\distributed_models"
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


def tuple_problem():

    class dummy(nn.Module):
        def __init__(self, first=True):
            super(dummy, self).__init__()
            self.layer = nn.Linear(10, 10)
            self.first = first

        def forward(self, *xs):
            if isinstance(xs[0], tuple):
                assert len(xs) == 1 and len(xs[0]) == 2
                return self.t_forward(xs[0])

            assert len(xs) == 2
            return self.m_forward(*xs)

        def t_forward(self, xs):
            x0, x1 = xs
            return self.m_forward(x0, x1)

        def m_forward(self, x0, x1):
            if self.first:
                return self.layer(x0), x1+1e-5
            return x0+1e-5, self.layer(x1)

    class seqDummy(nn.Module):
        def __init__(self, tupled):
            super(seqDummy, self).__init__()
            self.tupled = tupled

            self.t0 = dummy()
            self.t1 = dummy(first=False)

        def forward(self, *xs):
            if self.tupled:
                assert len(xs) == 1 and len(xs[0]) == 2
                return self.t_forward(xs[0])

            assert len(xs) == 2
            return self.m_forward(*xs)

        def t_forward(self, xs):
            a = self.t0(xs)
            return self.t1(a)

        def m_forward(self, x0, x1):
            a, b = self.t0(x0, x1)
            return self.t1(a, b)

    tupled = seqDummy(True)
    multi = seqDummy(False)
    sample = (torch.zeros(4, 10), torch.zeros(10, 10))
    g1 = partition_with_profiler(
        tupled, sample, nparts=2)

    g2 = partition_with_profiler(
        multi, *sample, nparts=2)

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = f"{curr_dir}\\partition_visualization"

    g1.save(directory=out_dir, file_name="tupled",
            show_buffs_params=False, show_weights=False)
    g2.save(directory=out_dir, file_name="multi",
            show_buffs_params=False, show_weights=False)

    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(
            tupled, (sample,))
        tupled_trace = trace_graph.graph()

        trace_graph, _ = torch.jit.get_trace_graph(
            multi, sample)
        multi_trace = trace_graph.graph()

    print(tupled_trace)

    print("\n")
    print(multi_trace)


if __name__ == "__main__":
    distribute_torchvision(networks=my_amoeaba, nparts=2,
                           save_graph=False, fake_gpus=True, nruns=1)
