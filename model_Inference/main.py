# %%
import torch.nn as nn

import torch

from network_profiler import NetProfiler

from res_net_example import resnet20_cifar
# hiddenlayer.build_graph(resnet20_cifar(), torch.zeros(1, 3, 32, 32))


class complexNet(nn.Module):
    def __init__(self):
        super(complexNet, self).__init__()
        a = nn.Linear(2, 2)

        self.sub1 = nn.Sequential(
            nn.Sequential(a),
            a, nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))

        self.sub2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.sub2(self.sub1(x))


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = resnet20_cifar()
    profiler = NetProfiler(complexNet())
    profiler.to(device)
    output = profiler(torch.tensor([[1.0, 2.0], [2.0, 2.0]]).to(device))
