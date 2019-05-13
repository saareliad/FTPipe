# %%
import torch.nn as nn
import torch
import inspect
from model_Inference.network_profiler import profileNetwork
from model_Inference.control_flow_graph import build_control_flow_graph
from model_Inference.graph_partition import part_graph
from pprint import pprint
from model_Inference.res_net_example import resnet20_cifar
from IPython.core.display import display_svg


def partition_model(model, num_gpus, *sample_batch, num_iter=2, max_depth=100, basic_blocks=None, device="cuda", weights=None, wrappers=None):

    if weights is None:
        weights = profileNetwork(model, *sample_batch, max_depth=max_depth,
                                 basic_block=basic_blocks, device=device, num_iter=num_iter)

    graph = build_control_flow_graph(
        model, *sample_batch, max_depth=max_depth, weights=weights, basic_block=basic_blocks, device=device)

    adjlist = graph.adjacency_list()
    nodew = graph.get_weights()
    weights = []
    for w in nodew:
        if isinstance(w, tuple):
            weights.append(int(w.forward_time))
        else:
            weights.append(int(w))

    cuts, parts = part_graph(
        adjlist, nparts=num_gpus, algorithm="metis", nodew=weights, contig=1)

    graph.set_partition(parts)

    return graph, cuts, parts


class complex_model(nn.Module):
    def __init__(self):
        super(complex_model, self).__init__()
        self.register_buffer("buff", torch.ones(1, 1))
        self.sub_1 = branched_model()
        self.sub_2 = branched_model()

    def forward(self, x, y):
        return self.sub_1(x) * self.buff, self.sub_2(y) + self.buff


class branched_model(nn.Module):
    def __init__(self):
        super(branched_model, self).__init__()

        # the branches can be a more complicated chains
        self.branch_1 = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1), nn.Linear(1, 1))
        self.branch_2 = nn.Sequential(nn.Sigmoid())

        self.relu = nn.ReLU()

    def forward(self, x):

        # we cannot determine if branches are independent or not
        branch_1_out = self.branch_1(x)*5
        branch_2_out = self.branch_2(x)*7

        # combine branches
        # our partition must put both branch outputs on the same device if not an error would occur
        # another problem is that an operation like this is "invisible" to us
        combined_branches = branch_1_out + branch_2_out

        # we cannot know that relu input is the combined output of both layers
        out = self.relu(combined_branches)

        return out


# infer control flow via scope name we can get it from the model and from the trace graph
# thus we can walk the graph and the names will tell us if we have a route from layer to layer
# names can be obtained easily and they represent scope (depth) and
model = resnet20_cifar()
graph, cuts, parts = partition_model(
    model, 4, torch.zeros(1, 3, 32, 32), max_depth=100)

graph.display()
