import torch.nn as nn
import torch
import pathlib
import os
from pytorch_Gpipe import graph_builder, generatePartitionModules


class Tuples(nn.Module):
    def __init__(self):
        super(Tuples, self).__init__()
        self.l0 = TupleOut()
        self.l1 = TupleOut()
        self.l2 = TupleOut()

    def forward(self, x):
        a, b = self.l0(x, x)

        t = self.l1(b, a)

        return self.l2(*t)


class TupleOut(nn.Module):
    def __init__(self):
        super(TupleOut, self).__init__()
        self.l0 = nn.ReLU()
        self.l1 = nn.Sigmoid()

    def forward(self, x, y):
        x = self.l0(x)
        y = self.l1(y)

        y *= 2
        x *= 3

        return y, x


def generate_graph(model, sample, save_jit_trace=False, generate_code=False, save_graph=False, output_path="jit_trace_bug"):
    name = model.__class__.__name__

    traced = torch.jit.trace(model, sample, check_trace=False)
    if save_jit_trace:
        clear_file(f"{output_path}/jit_trace/{name}.txt")
        with open(f"{output_path}/jit_trace/{name}.txt", "w") as f:
            f.write(str(traced.graph))

    graph = graph_builder(model, sample, max_depth=0)
    if save_graph:
        graph.save(name, f"{output_path}/graphs",
                   show_buffs_params=True, show_weights=False)

    if generate_code:
        generatePartitionModules(graph, model, verbose=True,
                                 output_file=f"{output_path}/generated/{name}")

    for node in traced.graph.nodes():
        if node.scopeName() == "":
            print(
                f"the node {node}does not have a scopeName that's a bug causing incorrect code generation")


def clear_file(path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.remove(path)


if __name__ == "__main__":
    model = Tuples()
    sample = torch.randn(100, 100)
    generate_graph(model, sample, save_jit_trace=True,
                   generate_code=True, save_graph=True)
