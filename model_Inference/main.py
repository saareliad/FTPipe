import torch.nn as nn

import torch
import model_Inference.partition_graph as pg
import numpy as np

"""""
from network_profiler import NetProfiler

from res_net_example import resnet20_cifar, BasicBlock

# hiddenlayer.build_graph(resnet20_cifar(), torch.zeros(1, 3, 32, 32))

"""
if __name__ == "__main__":
    """""
    base_model = resnet20_cifar()
    test_model = resnet20_cifar()

    base_profiler = NetProfiler(
        base_model, torch.randn(128, 3, 32, 32))

    test_profiler = NetProfiler(test_model, torch.rand(
        128, 3, 32, 32), basic_block=BasicBlock)
"""
    mat = [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],[0, 0, 0, 0, 0]]
    mat = np.matrix(mat)
    nodes_values = np.array([1, 2, 3, 4, 2])
    print(pg.partition_graphs(mat, nodes_values, 2))
