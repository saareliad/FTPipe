from model_Inference.network_profiler import NetProfiler
from model_Inference.res_net_example import resnet20_cifar
import torch.nn as nn
from models import AlexNet
import torch
import inspect
from pprint import pprint
import re


# scope names of all profiled layers in the model
def scopeNames(module: nn.Module, depth, prefix, basic_block=None):

    for name, sub_module in module._modules.items():
        # assume no cyclic routes in the network
        # a module with no children is a layer
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            print((prefix+"/"+name)[1:])
        else:
            scopeNames(sub_module, depth-1, prefix+"/"+name)


# print 3 types of graph raw ,raw optimized, ONNX optimized
def compare_graphs():
    model = branched_model()

    optimized_raw_graph = trace_graph(model, torch.zeros(1, 1),
                                      op_type=torch.onnx.OperatorExportTypes.RAW)

    print("optimized_raw_graph")
    print(optimized_raw_graph)
    print("\n\n\n")

    raw_graph = trace_graph(model, torch.zeros(1, 1), optimized=False,
                            op_type=torch.onnx.OperatorExportTypes.RAW)

    print("raw_graph")
    print(raw_graph)
    print("\n\n\n")

    optimized_ONNX_graph = trace_graph(model, torch.zeros(1, 1))

    print("optimized_ONNX_graph")
    print(optimized_ONNX_graph)
    print("\n\n\n")


# return a trace graph of a model convenience method
def trace_graph(model, *sample_inputs, optimized=True, op_type=torch.onnx.OperatorExportTypes.ONNX):
    trace_graph, _ = torch.jit.get_trace_graph(model, sample_inputs)
    if optimized:
        trace_graph.set_graph(_optimize_graph(trace_graph.graph(), op_type))

    trace_graph = trace_graph.graph()

    return trace_graph


# optimizes a graph using gives op type
# a copy of torch.onnx.utils._optimize_graph
def _optimize_graph(graph, operator_export_type):
    # TODO there is a bug with the second scope name of sequential is carried to all layers after it in the sequence
    # maybe can be fixed bug is in torch/onnx/utils.py/139
    from torch.onnx.utils import _split_tensor_list_constants, OperatorExportTypes
    torch._C._jit_pass_remove_inplace_ops(graph)
    # we record now record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    torch._C._jit_pass_constant_propagation(graph)
    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    # torch._C._jit_pass_canonicalize_ops(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)
    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)
    # onnx does not support tuples, so try to remove them
    torch._C._jit_pass_lower_all_tuples(graph)
    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    if operator_export_type != OperatorExportTypes.RAW:
        graph = torch._C._jit_pass_onnx(graph, operator_export_type)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_onnx_peephole(graph)
        torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_fixup_onnx_loops(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


# build our graph from a trace graph
def build_exec_graph(graph, num_inputs):
    total_num_nodes = 0
    # inputs params buffs
    for i, node in enumerate(graph.inputs()):
        # ignore unsued inputs / parameters / buffers
        if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
            continue

        total_num_nodes += 1

        if i < num_inputs:
            print("input node")
        else:

            print("parameter or buffer")

    # network ops
    for node in graph.nodes():
        outputs = [o.unique() for o in node.outputs()]
        inputs = [i.unique() for i in node.inputs()]
        print(f"node number is {total_num_nodes}")
        print(f"inputs are {inputs}")
        print(f"outputs are {outputs}")
        print(f"node's scopeName {node.scopeName()}")
        total_num_nodes += 1
    #     nodes_py.append(NodePyOP(node))

    # network outputs
    num_outs = 0
    for output in graph.outputs():
        print(output.unique())
        num_outs += 1


class complex_model(nn.Module):
    def __init__(self):
        super(complex_model, self).__init__()
        self.register_buffer("buff", torch.ones(1, 1))
        self.sub_1 = branched_model()
        self.sub_2 = branched_model()

    def forward(self, x, y):
        return self.sub_1(x) * self.buff, self.sub_2(y) + self.buff


class branched_model(nn.Module):
    def __init__(self):
        super(branched_model, self).__init__()

        # the branches can be a more complicated chains
        self.branch_1 = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1), nn.Linear(1, 1))
        self.branch_2 = nn.Sequential(nn.Sigmoid())

        self.relu = nn.ReLU()

    def forward(self, x):

        # we cannot determine if branches are independent or not
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)

        # combine branches
        # our partition must put both branch outputs on the same device if not an error would occur
        # another problem is that an operation like this is "invisible" to us
        combined_branches = branch_1_out + branch_2_out

        # we cannot know that relu input is the combined output of both layers
        out = self.relu(combined_branches)*2

        return out


# infer control flow via scope name we can get it from the model and from the trace graph
# thus we can walk the graph and the names will tell us if we have a route from layer to layer
# names can be obtained easily and they represent scope (depth) and
if __name__ == "__main__":
    # type(torch_graph)
    # # help(torch._C.Graph)
    # pprint(inspect.getmembers(torch._C.Graph))
    # print("\n\n")
    # pprint(inspect.getmembers(torch._C.Node))
    # # print(torch_graph)
    # print("done")
    # compare_graphs()
    model = complex_model()
    graph = trace_graph(
        model, torch.zeros(1, 1), torch.ones(1, 1))
    print(graph)
    build_exec_graph(graph, 2)
    # # print(help(torch._C.Graph))
    # pprint(inspect.getmembers(torch._C.Graph))
    # print(help(torch._C.Graph.at))
    # pprint(help(torch._C.Value))
