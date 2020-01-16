import torch.nn as nn
import torch
import pathlib
import os
import inspect
from itertools import chain


class Tuples(nn.Module):
    def __init__(self):
        super(Tuples, self).__init__()
        self.l0 = TupleOut()
        self.l1 = TupleOut()

    def forward(self, x):
        t = self.l0(x, x)

        return self.l1(*t)


class TupleOut(nn.Module):
    def __init__(self):
        super(TupleOut, self).__init__()
        self.l0 = Act(nn.ReLU())
        self.l1 = Act(nn.Sigmoid())

    def forward(self, x, y):
        x = self.l0(x)
        y = self.l1(y)

        y *= 2
        x *= 3

        return y, x


class Act(nn.Module):
    def __init__(self, act):
        super(Act, self).__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)*5


def generate_graph(model, sample, save_jit_trace=False, output_path="jit_trace_bug"):
    name = model.__class__.__name__
    torch._C._jit_set_inline_everything_mode(True)
    verbose = torch.jit.trace(model, sample, check_trace=False).graph
    torch._C._jit_set_inline_everything_mode(False)
    minimal = torch.jit.trace(model, sample, check_trace=False).graph

    if save_jit_trace:
        clear_file(f"{output_path}/{name}.txt")
        with open(f"{output_path}/{name}.txt", "w") as f:
            f.write(str(minimal))
            f.write("\n")
            f.write(str(verbose))

    if any(node.scopeName() != "" for node in verbose.nodes()):
        print("verbose trace has scopes")
    else:
        print("verbose trace des not have scopes")

    if any(node.scopeName() != "" for node in minimal.nodes()):
        print("minimal trace has scopes")
    else:
        print("minimal trace des not have scopes")


def clear_file(path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.remove(path)


if __name__ == "__main__":
    model = Tuples()
    sample = torch.randn(100, 100)
    generate_graph(model, sample, save_jit_trace=True)
