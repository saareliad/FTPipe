import torch.nn as nn

import torch
import numpy as np


from network_profiler import NetProfiler

from res_net_example import resnet20_cifar, BasicBlock

# hiddenlayer.build_graph(resnet20_cifar(), torch.zeros(1, 3, 32, 32))


if __name__ == "__main__":

    base_model = resnet20_cifar()

    base_profiler = NetProfiler(base_model, torch.randn(128, 3, 32, 32))

    print("done")
