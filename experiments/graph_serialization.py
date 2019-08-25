import pickle
from pytorch_Gpipe.model_profiling.control_flow_graph import Graph
import torch
from pytorch_Gpipe import partition_with_profiler
from sample_models.amoebaNet import AmoebaNet_D
import sys


def serialize_graph(graph: Graph, file_name):
    sys.setrecursionlimit(10000)
    pickle.dump(graph, open(file_name, "wb"))


def deserialize_graph(file_name):
    graph = pickle.load(open(file_name, "rb"))
    return graph


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    net = AmoebaNet_D()
    g = partition_with_profiler(net, x, nparts=4, max_depth=100, basic_blocks=None)

    serialize_graph(g)
    g_tag = deserialize_graph()

    print(g_tag)
    print("##########################################################################################################")
    print(g)
