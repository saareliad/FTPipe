
import torch.nn as nn
import torch
from enum import Enum
from pprint import pprint


def build_control_flow_graph(model, weights, *sample_batch, max_depth=None, basic_block=None):
    max_depth = max_depth if max_depth != None else 100
    layerNames = _profiled_layers(model, max_depth, prefix=type(
        model).__name__, basic_block=basic_block)

    model_trace = trace_graph(model, *sample_batch)

    return Graph(layerNames, weights, model_trace)


class NodeTypes(Enum):
    IN = 1
    LAYER = 2
    OP = 3

    def __repr__(self):
        return self.name


class Node():
    def __init__(self, scope, idx, node_type: NodeTypes, incoming_edges=None, weight=0):
        self.scope = scope
        self.idx = {idx}
        self.type = node_type
        self.out_edges = set()
        self.weight = weight
        self.in_edges = incoming_edges if isinstance(
            incoming_edges, set) else set()

    def add_outgoing_edge(self, edge):
        self.out_edges.add(edge)

    def __repr__(self):
        return f"node {self.idx} in scope {self.scope} of type {self.type} flows to {self.out_edges} gathers {self.in_edges}\n"


class Graph():
    def __init__(self, profiled_layers, weights, trace_graph):
        self.adjacency_list = []
        self.profiled_layers = profiled_layers
        self.weights = dict()
        self.num_inputs_buffs_params = 0
        self._build_graph(trace_graph)

    def _build_graph(self, trace_graph):
        self._add_IO_nodes(trace_graph.inputs())
        self._add_OP_nodes(trace_graph.nodes())
        self._combine_nodes_under_the_same_scope()
        self._remove_constant_nodes()

    def _add_IO_nodes(self, input_nodes):
        for node in input_nodes:
            node_scope = f"input{self.num_inputs_buffs_params}"
            new_node = Node(node_scope, self.num_inputs_buffs_params,
                            NodeTypes.IN)
            self.adjacency_list.append(new_node)

            self.weights[node.unique()] = node.type().sizes()
            self.num_inputs_buffs_params += 1

    def _add_OP_nodes(self, OP_nodes):
        for idx, trace_node in enumerate(OP_nodes):
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            inputs = {i.unique() for i in trace_node.inputs()}
            node_idx = self.num_inputs_buffs_params+idx

            # add incoming edges
            for in_idx in inputs:
                self.adjacency_list[in_idx].add_outgoing_edge(node_idx)

            # TODO add weights
            # profiled Layer
            if node_scope != "":
                self.adjacency_list.append(
                    Node(node_scope, node_idx, NodeTypes.LAYER, inputs))

            # unprofiled OP
            else:
                node_scope = trace_node.scopeName() + \
                    "/"+trace_node.kind() + str(idx)
                self.adjacency_list.append(
                    Node(node_scope, node_idx, NodeTypes.OP, inputs))

    def _find_encasing_layer(self, scopeName: str):
        '''
        longest prefix is the most specific layer that was profiled
        '''
        # unfortunately the trace graph shows only basic layers and ops
        # so we need to manually find a profiled layer that encases the op
        most_specific_scope = ""
        for layer_scope in self.profiled_layers:
            if scopeName.startswith(layer_scope) and len(layer_scope) > len(most_specific_scope):
                most_specific_scope = layer_scope

        return most_specific_scope

    def _combine_nodes_under_the_same_scope(self):
        # TODO build optimized graph in 2 iterations
        first_Node_of_scope = dict()
        optimized_graph = []

        idx = 0
        # get the nodes of the optimized graph
        for node in self.adjacency_list:
            if not node.scope in first_Node_of_scope:
                node.idx = idx
                optimized_graph.append(node)
                idx += 1
                first_Node_of_scope[node.scope] = node

            else:
                # add edges create the subset of all  edeges of the scope
                first_Node_of_scope[node.scope].in_edges.update(node.in_edges)
                first_Node_of_scope[node.scope].out_edges.update(
                    node.out_edges)

        for node in optimized_graph:
            # incoming and outgoing scopes are the sets of in/out scopes
            incoming_scopes = {self.adjacency_list[idx].scope
                               for idx in node.in_edges}
            incoming_scopes.discard(node.scope)

            outgoing_scopes = {self.adjacency_list[idx].scope
                               for idx in node.out_edges}
            outgoing_scopes.discard(node.scope)

            # update the edges of the new graph
            node.in_edges = {first_Node_of_scope[scope].idx
                             for scope in incoming_scopes}
            node.out_edges = {first_Node_of_scope[scope].idx
                              for scope in outgoing_scopes}

        self.adjacency_list = optimized_graph

    def _remove_constant_nodes(self):
        # optimized_graph = []
        # constant_nodes = set()
        # for node in self.adjacency_list:
        #     if "::Constant" in node.scope:
        #         constant_nodes.add(node.idx)
        #     else:
        #         optimized_graph.append(node)

        # for node in optimized_graph:
        #     node.in_edges.difference_update(constant_nodes)

        # self.adjacency_list = optimized_graph
        return False

    def __getitem__(self, key):
        return self.adjacency_list[key]

    def partition(self, num_parts):
        return


# ONNX_ATEN RAW
# return a trace graph of a model convenience method
def trace_graph(model, *sample_inputs, optimized=True, op_type=torch.onnx.OperatorExportTypes.RAW):
    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(model, sample_inputs)
        if optimized:
            trace_graph.set_graph(_optimize_graph(
                trace_graph.graph(), op_type))

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
