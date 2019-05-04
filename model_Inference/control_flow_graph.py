
import torch.nn as nn
import torch


def build_control_flow_graph(model, weights, *sample_batch, max_depth=None, basic_block=None):
    max_depth = max_depth if max_depth != None else 100
    layerNames = _profiled_layers(model, max_depth, prefix=type(
        model).__name__, basic_block=basic_block)

    model_trace = trace_graph(model, *sample_batch)

    return Graph(layerNames, weights, model_trace)


class Graph():
    def __init__(self, profiled_layers, weights, trace_graph):
        self.adjacency_list = []
        self.profiled_layers = profiled_layers
        self.weights = dict()
        self.num_inputs_buffs_params = 0
        self.num_op_nodes = 0

        self.add_IO_nodes(trace_graph.inputs())

        self.add_OP_nodes(trace_graph.nodes())

    def add_IO_nodes(self, input_nodes):
        for node in input_nodes:
            # add input nodes only if they are in use
            # TODO if they are not in use how will it affect the other nodes
            if len(node.uses()) > 0:
                self.adjacency_list.append([])
                self.weights[node.unique()] = node.type().sizes()
                self.num_inputs_buffs_params += 1

    def add_OP_nodes(self, OP_nodes):
        for trace_node in OP_nodes:
            node_scope = self.find_encasing_layer(trace_node.scopeName())

            outputs = [o.unique() for o in trace_node.outputs()]
            inputs = [i.unique() for i in trace_node.inputs()]

            # add incoming edges from inputs as they do not have an output field
            for in_idx in inputs:
                if in_idx < self.num_inputs_buffs_params:
                    self.adjacency_list[in_idx].append(
                        self.num_inputs_buffs_params+self.num_op_nodes)

            # TODO multiple outputs and self idx possibly only if we combine nodes
            self.adjacency_list.append(list(set(outputs)))

            # TODO add weights
            if node_scope != "":
                # self.weights[trace_node.unique()] = weights[node_scope]
                print("layer")
            else:
                print("arithmetic")

            self.num_op_nodes += 1

    def find_encasing_layer(self, scopeName: str):
        '''
        longest prefix is the most specific layer that was profiled
        '''
        # unfortunately the trace graph shows only basic layers and ops
        # so we need to manually find a profiled layer that encases the op
        most_specific_scope = ""
        print(scopeName)
        for layer_scope in self.profiled_layers:
            if scopeName.startswith(layer_scope) and len(layer_scope) > len(most_specific_scope):
                most_specific_scope = layer_scope

        return most_specific_scope

    def __getitem__(self, key):
        return self.adjacency_list[key], self.weights.get(key, 0)

    def partition(self, num_parts):
        return


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


# scope names of all profiled layers in the model
def _profiled_layers(module: nn.Module, depth, prefix='', basic_block=None):
    names = []
    for name, sub_module in module._modules.items():
        # assume no cyclic routes in the network
        # a module with no children is a layer
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            names.append(
                prefix+"/"+type(sub_module).__name__+f"[{name}]")
        else:
            names = names + _profiled_layers(sub_module, depth-1, prefix +
                                             "/"+type(sub_module).__name__+f"[{name}]")
    return names
