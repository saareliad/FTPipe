# %%
import torch.nn as nn

import torch
import torchvision
from network_profiler import NetProfiler

from res_net_example import resnet20_cifar
# hiddenlayer.build_graph(resnet20_cifar(), torch.zeros(1, 3, 32, 32))


class complexNet(nn.Module):
    def __init__(self):
        super(complexNet, self).__init__()

        self.sub1 = nn.Sequential(
            nn.Sequential(nn.Linear(2, 10)),
            nn.Linear(10, 2), nn.ReLU(), nn.Linear(2, 4), nn.Sequential(nn.Linear(4, 2)))

        self.sub2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.sub2(self.sub1(x))


if __name__ == "__main__":

    base_model = resnet20_cifar()
    profiler = NetProfiler(base_model, torch.randn(3, 3, 32, 32))
    torch.cuda.synchronize()
    print("done")
