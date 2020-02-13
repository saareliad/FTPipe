import builtins
import inspect
import os
import pathlib
import re
import string
import time
from collections import OrderedDict
from datetime import datetime
from itertools import chain, permutations
from pprint import pprint
from typing import Generator, List

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, jit, nn, no_grad, randn
from torch.nn import Parameter, TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.linear import Linear

from misc import run_analysis
from pytorch_Gpipe import (METIS_partition, Pipeline, build_graph,
                           compile_partitoned_model,
                           pipe_model, profile_network)
from pytorch_Gpipe.model_profiling import NodeTypes
from pytorch_Gpipe.model_profiling.control_flow_graph import Graph, Node
from pytorch_Gpipe.model_profiling.graph_builder import translate_scopes
from pytorch_Gpipe.utils import (OrderedSet, layerDict, tensorDict,
                                 traverse_model, traverse_params_buffs)
import sample_models
from partitioned_models.tmp.resnet18_2p import create_pipeline_configuration, ResNetModelParallel


def node_weight_function(node: Node):
    if node.type is NodeTypes.LAYER:
        return int(node.weight.forward_time + node.weight.backward_time)
    if node.type is NodeTypes.CONSTANT:
        return 0
    if node.type is NodeTypes.OP:
        return 1
    return 0


def edge_weight_function(bandwidth_gps=10):
    def f(u: Node, v: Node):
        if u.type is NodeTypes.LAYER:
            return max(1, int(u.weight.output_size / bandwidth_gps))
        if v.type is NodeTypes.LAYER:
            return max(1, int(v.weight.input_size / bandwidth_gps))
        if u.type is NodeTypes.CONSTANT:
            return 1000
        return 1
    return f


class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            # nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(False),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            # nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(False),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            # nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(False),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            # nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(False),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            # nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(False),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            # nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(False),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            # nn.BatchNorm2d(pool_planes),
            nn.ReLU(False),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y = torch.cat([y1, y2, y3, y4], 1)

        return y


def err(expected, actual):
    print(torch.allclose(expected, actual, atol=1e-7))
    print("avg abs error", (expected - actual).abs().sum() / expected.numel())
    print("avg squared error", ((expected - actual)**2).sum() / expected.numel())
    print()


def check_inception():
    batch_size = 512
    num_chunks = 4
    num_batches = 5
    sample = torch.randn(batch_size, 192, 32, 32).cuda()
    model = Inception(192, 64, 96, 128, 16, 32, 32).cuda()

    graph = pipe_model(model, sample, node_weight_function=node_weight_function,
                       edge_weight_function=edge_weight_function(10), n_iter=50, output_file="inception")
    graph.save_as_pdf("inception", ".")
    # get expected results
    torch.manual_seed(0)
    expected = model(sample)
    model(sample).sum().backward()
    expected_gs = sorted([(n, p.grad.clone().cpu())
                          for n, p in model.named_parameters()])
    expected_gs = [p for n, p in expected_gs]
    model.zero_grad()

    sample = sample.clone().cpu()
    expected = expected.clone().cpu()
    from inception import create_pipeline_configuration
    # return
    configs = create_pipeline_configuration(model, DEBUG=True)
    pipe = Pipeline(configs, output_device='cpu', use_delayedNorm=False)
    for _ in range(num_batches):
        pipe.zero_grad()
        torch.manual_seed(0)
        actual = pipe(sample, num_chunks=num_chunks)
        err(expected, actual)
        # check backward
        pipe.backward(torch.autograd.grad(actual.sum(), [actual]))
        actual_gs = sorted([(n, p.grad.clone())
                            for n, p in pipe.named_parameters()])
        actual_gs = [p for n, p in actual_gs]

        for e, a in zip(expected_gs, actual_gs):
            err(e, a)


# tests and edge cases
def inplace_bug(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class InplaceBug(nn.Module):
        def __init__(self):
            super(InplaceBug, self).__init__()
            self.linear0 = nn.Linear(100, 100)
            self.linear1 = nn.Linear(100, 100)
            self.w = Parameter(torch.randn(100).requires_grad_())
            self.register_buffer("b", torch.randn(100))

        def forward(self, x):
            x = self.linear0(x)
            x = self.linear1(x)
            x[:, 1] = 15
            return x + self.b + self.w
    model = InplaceBug()
    sample = torch.randn(100, 100)

    generate_graph(model, sample, "InplaceBug", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def unpack_bug(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class UnpackBug(nn.Module):
        def __init__(self):
            super(UnpackBug, self).__init__()
            self.l = nn.ReLU()

        def forward(self, x: Tensor):
            x = self.l(x)
            a, b, c = x.chunk(3, dim=-1)
            return a + 1, b + 2, c + 3

    model = UnpackBug()
    sample = torch.randn(10, 9)

    generate_graph(model, sample, "UnpackBug", depth=depth, trace_graph=trace_graph, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def moduleList_bug(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class ModuleList_bug(nn.Module):
        def __init__(self, i):
            super(ModuleList_bug, self).__init__()

            self.layers = nn.ModuleList(
                [nn.Linear(100, 100) for _ in range(i)])

        def forward(self, x):
            out = x
            for l in self.layers:
                out = l(out)
            return out

    class ModuleList_Hack(nn.Module):
        def __init__(self, i):
            super(ModuleList_Hack, self).__init__()

            for j in range(i):
                self.add_module(str(j), nn.Linear(100, 100))
            self.i = i

        def forward(self, x):
            out = x
            for j in range(self.i):
                out = getattr(self, str(j))(out)
            return out

    model, sample = ModuleList_bug(2), torch.randn(10, 100)

    model, sample = ModuleList_Hack(2), torch.randn(10, 100)
    generate_graph(model, sample, "ModuleListHack", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def multipleSameInput_bug(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class MultipleSameInput(nn.Module):
        def __init__(self):
            super(MultipleSameInput, self).__init__()
            self.l = nn.Sigmoid()
            self.a = A()

        def forward(self, x):
            return self.l(x), self.a(x, x)

    class A(nn.Module):
        def __init__(self):
            super(A, self).__init__()
            self.r = nn.ReLU()

        # if the same input is fed aka x is y then the trace will show only one input
        # assuming it's the same for output
        def forward(self, x, y):
            return x + 5, self.r(y * 2)

    model, sample = MultipleSameInput(), torch.randn(10, 100)

    generate_graph(model, sample, "MultipleSameInput", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def conversions_bug(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class TypeConversion(nn.Module):
        def __init__(self):
            super(TypeConversion, self).__init__()
            self.r = nn.ReLU()

        def forward(self, x):
            x = self.r(x)
            out = (x.to(torch.bool),
                   x.to(torch.long),
                   x.to(torch.int32),
                   x.to(torch.int16),
                   x.to(torch.uint8),
                   x.to(torch.int8),
                   x.to(torch.float16),
                   x.to(torch.float32),
                   x.to(torch.float64),
                   )
            return out

    class DeviceConversion(nn.Module):
        def __init__(self):
            super(DeviceConversion, self).__init__()
            self.r = nn.ReLU()

        def forward(self, x):
            x = self.r(x)
            out = (x.to('cuda'), x.to('cpu'),
                   x.to('cuda:0'), x.to('cpu', torch.bool),
                   x.to('cuda', torch.bool, non_blocking=True),
                   x.to('cuda:0', torch.bool, non_blocking=False, copy=True)
                   )
            return out

    a = DeviceConversion()
    generate_graph(a, torch.randn(10, 10), "DeviceConversion", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)

    a = TypeConversion()
    generate_graph(a, torch.randn(10, 10), "TypeConversion", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def checkKeywordInput(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class keywordInput(nn.Module):
        def __init__(self):
            super(keywordInput, self).__init__()
            self.l0 = nn.Linear(20, 100)
            self.l1 = nn.Linear(20, 100)

        def forward(self, d):
            x0 = d['x0']
            x1 = d.get('x1', None)
            if x1 is None:
                x1 = torch.ones_like(x0)
            return self.l0(x0) + self.l1(x1)

    a = keywordInput()
    kwargs = {'x1': torch.randn(50, 20), 'x0': torch.randn(50, 20)}
    for k in kwargs:
        print(k)
    args = (torch.randn(50, 20),)
    # TODO pytorch do not support kwargs and tracing
    generate_graph(a, (), "KeywordInput", kwargs=kwargs, trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def checkArange(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class Arange(nn.Module):
        def __init__(self):
            super(Arange, self).__init__()
            self.l = nn.ReLU()

        def forward(self, x):
            # 7 7 5
            return self.l(x), torch.arange(0, 3, 1) + torch.arange(2, 3) + torch.arange(3)

    a = Arange()

    generate_graph(a, torch.randn(100, 100), "Arange", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def checkSliceSize(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    class sliceSize(nn.Module):
        def __init__(self):
            super(sliceSize, self).__init__()
            self.l = nn.ReLU()

        def forward(self, x):
            x[:, 1] += 2
            size = self.l(x).shape[:-1]
            size = size + (2, -1)
            return x.view(size)

    a = sliceSize()
    sample = torch.rand(10, 3, 224, 224)

    generate_graph(a, sample, "SliceSize", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def checkStringArg(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    # the reduction is coded as an int
    class StringArg(nn.Module):
        def __init__(self):
            super(StringArg, self).__init__()
            self.l = nn.LogSoftmax(dim=1)

        def forward(self, x, y):
            logits = self.l(x)
            return F.nll_loss(logits, y, reduction='none'), F.nll_loss(logits, y, reduction='sum'), F.nll_loss(logits, y, reduction='mean')

    x = torch.randn(10, 4)
    y = torch.randint(4, (10,))

    a = StringArg()

    generate_graph(a, (x, y), "StringArg", trace_graph=trace_graph, depth=depth, minimal=minimal,
                   jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def checkTuples(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):

    class Tuples(nn.Module):
        def __init__(self):
            super(Tuples, self).__init__()
            self.l0 = TupleOut()
            self.l1 = TupleOut()
            self.linear = nn.Linear(50, 50)

        def forward(self, x):
            x = self.linear(x)
            x *= 13
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

    model = Tuples()
    sample = torch.randn(100, 50)

    return generate_graph(model, sample, "Tuples", trace_graph=trace_graph, depth=depth, minimal=minimal, basic_blocks=None,
                          jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def checkNested(depth=1000, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):

    class A(nn.Module):
        def __init__(self):
            super(A, self).__init__()
            self.l0 = Nested()
            self.l1 = Nested()
            self.linear = nn.Linear(50, 50)

        def forward(self, x):
            x = self.linear(x)
            x *= 13
            t = self.l0(x)
            return self.l1(t)

    class Nested(nn.Module):
        def __init__(self):
            super(Nested, self).__init__()
            self.l0 = Act(nn.ReLU())
            self.l1 = Act(nn.Sigmoid())

        def forward(self, x):
            x = self.l0(x)
            y = self.l1(x)

            y *= 2
            x *= 3

            return x + y

    class Act(nn.Module):
        def __init__(self, act):
            super(Act, self).__init__()
            self.act = act

        def forward(self, x):
            return self.act(x) * 5

    model = A()
    sample = torch.randn(100, 50)

    return generate_graph(model, sample, "checkNested", trace_graph=trace_graph, depth=depth, minimal=minimal, basic_blocks=None,
                          jit_trace=jit_trace, scripted=scripted, generate_code=generate_code)


def generate_graph(model, sample, name, kwargs=None, depth=10000, basic_blocks=None, trace_graph=False, jit_trace=False, minimal=False, scripted=False, generate_code=False):
    if kwargs is None:
        kwargs = dict()
    if not isinstance(sample, tuple):
        sample = (sample,)
    if trace_graph:
        if hasattr(torch.jit, "get_trace_graph"):
            get_trace_graph = torch.jit.get_trace_graph
        else:
            assert hasattr(torch.jit, "_get_trace_graph")
            get_trace_graph = torch.jit._get_trace_graph
        trace_graph, _ = get_trace_graph(model, args=sample, kwargs=kwargs)

        clear_file(f"playground_out/get_trace/{name}.txt")
        with open(f"playground_out/get_trace/{name}.txt", "w") as f:
            f.write(str(trace_graph))
    if jit_trace:
        assert len(kwargs) == 0, "jit_trace does not support kwargs"
        old_value = torch._C._jit_get_inline_everything_mode()
        torch._C._jit_set_inline_everything_mode(not minimal)
        traced = torch.jit.trace(model, sample, check_trace=False)
        torch._C._jit_set_inline_everything_mode(old_value)
        clear_file(f"playground_out/jit_trace/{name}.txt")
        with open(f"playground_out/jit_trace/{name}.txt", "w") as f:
            f.write(str(traced.graph))
    if scripted:
        scripted_module = torch.jit.script(model)
        clear_file(f"playground_out/scripted/{name}.txt")
        with open(f"playground_out/scripted/{name}.txt", "w") as f:
            f.write(str(scripted_module.graph))

    graph = build_graph(model, sample, kwargs, depth, basic_blocks,
                        n_iter=10, )

    graph.save_as_pdf(name, f"playground_out/graphs/{name}",
                      show_buffs_params=True, show_weights=False)
    if generate_code:
        compile_partitoned_model(graph, model, verbose=False,
                                 output_file=f"playground_out/generated/{name}")


def clear_file(path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        os.remove(path)


def check_gnmt():
    model = sample_models.GNMT(512, hidden_size=512, num_layers=2,
                               share_embedding=False).cuda()

    sample = torch.randint(512, (512, 2)).cuda()

    # model(sample, torch.tensor(512).unsqueeze(0).cuda(), sample)

    build_graph(model, (sample, torch.tensor(512).unsqueeze(0).cuda(),
                        sample), n_iter=1).save_as_pdf("GNMT", ".")


class LLSTM(nn.Module):
    def __init__(self):
        super(LLSTM, self).__init__()
        self.l = nn.LSTM(512, 512, 2, batch_first=False, bidirectional=True)

    def forward(self, x):
        output, (hn, cn) = self.l(x)
        return output


def check_LSTM():
    rnn = LLSTM().cuda()
    sample = torch.randn(512, 1, 512).cuda()

    build_graph(rnn, sample).save_as_pdf("LLSTM", ".")


def pipeline_with_optimizer():
    model = sample_models.resnet18().cuda()
    sample = torch.randn(16, 3, 224, 224).cuda()

    config = create_pipeline_configuration(model, DEBUG=True)

    config[0]['optimizer'] = torch.optim.SGD(config[0]['model'].parameters(),
                                             lr=1e-3)
    config[1]['optimizer'] = torch.optim.SGD(config[1]['model'].parameters(),
                                             lr=1e-3)

    model = Pipeline(config, output_device='cpu', use_multiprocessing=True)

    out = model(sample.cpu(), num_chunks=4)
    model.backward(torch.ones_like(out))

    print(out.device)
    print("done")


def model_parallel():
    model = sample_models.resnet18().cuda()
    sample = torch.randn(16, 3, 224, 224).cuda()

    config = create_pipeline_configuration(model, DEBUG=True)
    config[0]['model'].cpu()
    config[1]['model'].cuda()
    assert config[1]['model'].device.type == 'cuda'
    model = ResNetModelParallel(config)

    output = model(sample)
    assert output.is_cuda
    print("done")

    for n, p in model.named_parameters():
        print(n, p.device)


if __name__ == "__main__":
    model_parallel()
