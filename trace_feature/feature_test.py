
import torch.nn as nn
import torch

import sys
sys.path.append("../")
from pytorch_Gpipe import build_graph, Graph, compile_partitoned_model
from pytorch_Gpipe.model_profiling.graph_builder import basic_blocks_new_scopes, layerDict, translate_scopes
import os


class Tuples(nn.Module):
    def __init__(self):
        super(Tuples, self).__init__()
        self.b1 = Branch1()
        self.b2 = Branch2()

    def forward(self, x):
        t = self.b1(x, x)

        return self.b2(*t)


class Branch1(nn.Module):
    def __init__(self):
        super(Branch1, self).__init__()
        self.l0 = Act(nn.ReLU(inplace=False))
        self.op = Act(nn.Linear(100, 100))

    def forward(self, x, y):
        x = self.l0(x)
        y = self.op(y)

        return y, x


class Branch2(nn.Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.l0 = Act(nn.ReLU(inplace=False))
        self.op = Branch1()

    def forward(self, x, y):
        x = self.l0(x)
        y = self.op(y, x)

        return y, x


class Act(nn.Module):
    def __init__(self, act):
        super(Act, self).__init__()
        self.act = act

    def forward(self, x):
        return self.act(x) * 5


# TODO need to recreate expected graphs as we now record shapes


def check_nothing_inlined(model, sample):
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    with open("actual_traces/minimal_trace.txt", "w") as f, open("expected_traces/minimal_trace.txt", "r") as e:
        f.write(str(traced))
        if str(traced) != e.read():
            print("check_nothing_inlined traces not equal")
            return False
    graph = build_graph(model, sample, max_depth=0)
    graph.serialize("actual_graphs/minimal_graph")

    if not Graph.deserialize("expected_graphs/minimal_graph").graphs_equal(graph):
        print("check_nothing_inlined graphs not equal")
        return False

    os.remove("actual_graphs/minimal_graph.graph")
    os.remove("actual_traces/minimal_trace.txt")

    return True


def check_we_only_inline_composite_layers(model, sample):
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    torch._C._jit_pass_inline(traced, depth=1000)
    with open("actual_traces/maximal_trace.txt", "w") as f, open("expected_traces/maximal_trace.txt", "r") as e:
        f.write(str(traced))
        if str(traced) != e.read():
            print("check_we_only_inline_composite_layers traces not equal")
            return False

    graph = build_graph(model, sample, max_depth=1000)
    graph.serialize("actual_graphs/maximal_graph")

    if not Graph.deserialize("expected_graphs/maximal_graph").graphs_equal(graph):
        print("check_we_only_inline_composite_layers graphs not equal")
        return False

    os.remove("actual_graphs/maximal_graph.graph")
    os.remove("actual_traces/maximal_trace.txt")

    return True


def check_depth_based_tracing(model, sample):
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    torch._C._jit_pass_inline(traced, depth=1)
    with open("actual_traces/depth_1_trace.txt", "w") as f, open("expected_traces/depth_1_trace.txt", "r") as e:
        f.write(str(traced))
        if str(traced) != e.read():
            print("check_depth_based_tracing traces not equal")
            return False

    graph = build_graph(model, sample, max_depth=1)
    graph.serialize("actual_graphs/depth_1_graph")

    if not Graph.deserialize("expected_graphs/depth_1_graph").graphs_equal(graph):
        print("check_depth_based_tracing graphs not equal")
        return False

    os.remove("actual_graphs/depth_1_graph.graph")
    os.remove("actual_traces/depth_1_trace.txt")


def check_basic_blocks(model, sample):
    basic_blocks = (Branch1,)
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    # compute the scopes of the basic blocks
    profiled_layers = layerDict(model, depth=1000,
                                basic_blocks=basic_blocks)
    new_to_old = translate_scopes(profiled_layers.keys())
    block_scopes = basic_blocks_new_scopes(basic_blocks,
                                           profiled_layers, new_to_old)

    torch._C._jit_pass_inline(traced, depth=1000, basic_blocks=block_scopes)
    with open("actual_traces/basic_blocks_trace.txt", "w") as f, open("expected_traces/basic_blocks_trace.txt", "r") as e:
        f.write(str(traced))
        # the traces are not content equal as some ids are different
        if len(str(traced)) != len(e.read()):
            print("check_basic_blocks traces not equal")
            return False

    graph = build_graph(model, sample, max_depth=1000,
                        basic_blocks=basic_blocks)
    graph.serialize("actual_graphs/basic_blocks_graph")

    if not Graph.deserialize("expected_graphs/basic_blocks_graph").graphs_equal(graph):
        print("check_basic_blocks graphs not equal")
        return False

    os.remove("actual_graphs/basic_blocks_graph.graph")
    os.remove("actual_traces/basic_blocks_trace.txt")

    return True


def check_default_behaviour(model, sample):
    traced = torch.jit.trace(model, sample, check_trace=False).graph
    torch._C._jit_pass_inline(traced)
    with open("actual_traces/default_trace.txt", "w") as f, open("expected_traces/default_trace.txt", "r") as e:
        f.write(str(traced))
        calls = len([n for n in traced.nodes()
                     if n.kind() == "prim::CallMethod"])
        if calls != 0:
            print("check_default_behaviour traces not equal")
            return False

    os.remove("actual_traces/default_trace.txt")
    return True


if __name__ == "__main__":
    if not os.path.exists("actual_traces"):
        os.makedirs("actual_traces")
    if not os.path.exists("actual_graphs"):
        os.makedirs("actual_graphs")

    model = Tuples()
    sample = torch.randn(100, 100)

    check_nothing_inlined(model, sample)
    check_we_only_inline_composite_layers(model, sample)
    check_depth_based_tracing(model, sample)
    check_basic_blocks(model, sample)
    check_default_behaviour(model, sample)

    if not os.listdir("actual_graphs"):
        os.rmdir("actual_graphs")
    if not os.listdir("actual_traces"):
        os.rmdir("actual_traces")

    if (not os.path.exists("actual_traces")) and (not os.path.exists("actual_graphs")):
        print("feature works")
