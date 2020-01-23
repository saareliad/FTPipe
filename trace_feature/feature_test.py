
import torch.nn as nn
import torch

import sys
sys.path.append("../")
from pytorch_Gpipe import build_graph


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
        return self.act(x) * 5


if __name__ == "__main__":
    model = Tuples()
    sample = torch.randn(100, 100)

    traced = torch.jit.trace(model, sample, check_trace=False).graph

    # check nothing inlined
    calls = len([n for n in traced.nodes() if n.kind() == "prim::CallMethod"])
    assert calls == 2, f"expected 2 calls got {calls}"

    # check we inline TupleOut both in trace and graph
    torch._C._jit_pass_inline(traced, 1)

    calls = len([n for n in traced.nodes() if n.kind() == "prim::CallMethod"])
    assert calls == 4, f"expected 4 calls got {calls}"

    graph = build_graph(model, sample, max_depth=1)

    calss = len([n for n in graph.nodes if "Act" in n.scope])
    assert calls == 4, f"expected 4 graph nodes got {calls}"

    # check default parameter
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    torch._C._jit_pass_inline(traced)
    calls = len([n for n in traced.nodes() if n.kind() == "prim::CallMethod"])
    assert calls == 5, f"expected 5 calls got {calls}"
