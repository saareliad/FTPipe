
import torch.nn as nn
import torch

import sys
sys.path.append("../")
from pytorch_Gpipe import build_graph
from pytorch_Gpipe.model_profiling.graph_builder import basic_blocks_new_scopes, layerDict, translate_scopes


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
    torch._C._jit_pass_inline(traced, depth=1)

    calls = len([n for n in traced.nodes() if n.kind() == "prim::CallMethod"])
    assert calls == 4, f"expected 4 calls got {calls}"

    graph = build_graph(model, sample, max_depth=1)

    calss = len([n for n in graph.nodes if "Act" in n.scope])
    assert calls == 4, f"expected 4 graph nodes got {calls}"

    # check with basic block
    basic_blocks = (Act,)
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    # compute the scopes of the basic blocks
    profiled_layers = layerDict(model, depth=1000,
                                basic_blocks=basic_blocks)
    new_to_old = translate_scopes(profiled_layers.keys())
    block_scopes = basic_blocks_new_scopes(basic_blocks,
                                           profiled_layers, new_to_old)

    torch._C._jit_pass_inline(traced, depth=1000, basic_blocks=block_scopes)
    calls = len([n for n in traced.nodes() if n.kind() == "prim::CallMethod"])
    assert calls == 4, f"expected 4 calls got {calls}"

    graph = build_graph(model, sample, max_depth=1000,
                        basic_blocks=basic_blocks)
    calss = len([n for n in graph.nodes if "Act" in n.scope])
    assert calls == 4, f"expected 4 graph nodes got {calls}"

    # check default parameter
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    torch._C._jit_pass_inline(traced)
    calls = len([n for n in traced.nodes() if n.kind() == "prim::CallMethod"])
    assert calls == 0, f"expected 0 calls got {calls}"
