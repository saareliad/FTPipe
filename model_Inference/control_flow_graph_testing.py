import torch.nn as nn
import torch
from network_profiler import NetProfiler
from control_flow_graph import build_control_flow_graph
import inspect
from pprint import pprint
from res_net_example import resnet20_cifar


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
if __name__ == "__main__":
    model = resnet20_cifar()
    graph = build_control_flow_graph(
        model, 1, torch.zeros(1, 3, 32, 32), max_depth=100)
    pprint(graph.adjacency_list)
    # pprint(inspect.getmembers(torch._C.Value))
