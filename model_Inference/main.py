
import torch.nn as nn

import torch
from network_profiler import NetProfiler

from res_net_example import resnet20_cifar, BasicBlock
# hiddenlayer.build_graph(resnet20_cifar(), torch.zeros(1, 3, 32, 32))


if __name__ == "__main__":

    base_model = resnet20_cifar()
    test_model = resnet20_cifar()

    base_profiler = NetProfiler(
        base_model, torch.randn(128, 3, 32, 32))

    test_profiler = NetProfiler(test_model, torch.rand(
        128, 3, 32, 32), basic_block=BasicBlock)
