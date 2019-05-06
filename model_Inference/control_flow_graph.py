
import torch.nn as nn
import torch
from enum import Enum
from pprint import pprint

__all__ = ['build_control_flow_graph']


def build_control_flow_graph(model, weights, *sample_batch, max_depth=None, basic_block=None):
    max_depth = max_depth if max_depth != None else 100
    model_class_name = type(model).__name__
    layerNames = _profiled_layers(
        model, max_depth, prefix=model_class_name, basic_block=basic_block)

    model_trace = trace_graph(model, *sample_batch)

    return Graph(layerNames, weights, model_trace)


class NodeTypes(Enum):
    IN = 1
    LAYER = 2
    OP = 3

    def __repr__(self):
        return self.name


class Node():
    '''
    a simple graph node for directed graphs

    Fields:
    ------
    scope:
     the operation/layer the node represents
    idx:
     a serial number of the node for convience
    node_type:
     an enum representing if the node is an input Layer or operator(like arithmetic ops)
    incoming_nodes:
     the nodes who have edges from them to this node
    out_nodes:
     the nodes who have edges from this node

     parallel edges in the same direction are not allowed
    '''

    def __init__(self, scope, idx, node_type: NodeTypes, incoming_nodes=None, weight=0):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes = set()
        self.weight = weight
        self.in_nodes = incoming_nodes if isinstance(
            incoming_nodes, set) else set()

    def add_out_node(self, node):
        if isinstance(node, Node):
            self.out_nodes.add(node)
        if isinstance(node, set):
            self.out_nodes.update(node)

    def add_in_node(self, node):
        if isinstance(node, Node):
            self.in_nodes.add(node)
        if isinstance(node, set):
            self.in_nodes.update(node)

    def remove_in_node(self, node):
        if isinstance(node, Node):
            self.in_nodes.discard(node)
        if isinstance(node, set):
            self.in_nodes.difference_update(node)

    def remove_out_node(self, node):
        if isinstance(node, Node):
            self.out_nodes.discard(node)
        if isinstance(node, set):
            self.out_nodes.difference_update(node)

    def __repr__(self):
        out_idx = {node.idx for node in self.out_nodes}
        in_idx = {node.idx for node in self.in_nodes}
        return f"node {self.idx} in scope {self.scope} of type {self.type} flows to {out_idx} gathers {in_idx}\n"


class Graph():
    '''
    a graph representing the control flow of a model
    built from a pytorch trace graph.
    the graph vertices are specified using the given profiled layer names\n
    and will also include basic pytorch ops that connect them.
    the edges represent the data flow.
    '''

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
        self._merge_op_chains()

    def _add_IO_nodes(self, input_nodes):
        '''
        add nodes representing the input and params/buffs of the model
        '''
        for node in input_nodes:
            # TODO what if buffer is not used? remove or add
            # if len(list(node.uses())) == 0:
            #     continue
            node_scope = f"input{self.num_inputs_buffs_params}"
            new_node = Node(node_scope, self.num_inputs_buffs_params,
                            NodeTypes.IN)
            self.adjacency_list.append(new_node)

            self.weights[node.unique()] = node.type().sizes()
            self.num_inputs_buffs_params += 1

    def _add_OP_nodes(self, OP_nodes):
        '''
        add nodes representing the layers/ops of the model
        '''
        for idx, trace_node in enumerate(OP_nodes):
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            input_nodes = {self.adjacency_list[i.unique()]
                           for i in trace_node.inputs()}

            node_idx = self.num_inputs_buffs_params+idx
            new_node = None

            # TODO add weights
            # profiled Layer
            if node_scope != "":
                new_node = Node(node_scope, node_idx,
                                NodeTypes.LAYER, input_nodes)
            # unprofiled OP
            else:
                node_scope = trace_node.scopeName() + \
                    "/"+trace_node.kind() + str(idx)
                new_node = Node(node_scope, node_idx,
                                NodeTypes.OP, input_nodes)

            # add incoming edges
            for node in input_nodes:
                node.add_out_node(new_node)

            self.adjacency_list.append(new_node)

    def _find_encasing_layer(self, scopeName: str):
        '''
        find the closest scope which encases scopeName
        '''
        # unfortunately the trace graph shows only basic layers and ops
        # so we need to manually find a profiled layer that encases the op
        most_specific_scope = ""
        for layer_scope in self.profiled_layers:
            if scopeName.startswith(layer_scope) and len(layer_scope) > len(most_specific_scope):
                most_specific_scope = layer_scope

        return most_specific_scope

    def _combine_nodes_under_the_same_scope(self):
        # optimization that reduces number of nodes in the graph
        # combine nodes that have a commom scope we do this because\n
        # if nodes have the same scopeName than they were profiled together

        node_of_scope = dict()
        optimized_graph = []

        # get the nodes of the optimized graph
        for node in self.adjacency_list:
            if not node.scope in node_of_scope:
                optimized_graph.append(node)
                node_of_scope[node.scope] = node

            else:
                # add edges create the subset of all  edeges of the scope
                node_of_scope[node.scope].add_in_node(node.in_nodes)
                node_of_scope[node.scope].add_out_node(node.out_nodes)

        for node in optimized_graph:
            # get the sets of all incoming/outgoing scopes
            # those will dictate the new set of edges and
            # remove the internal edges of the scope
            incoming_scopes = {n.scope for n in node.in_nodes
                               if n.scope != node.scope}
            outgoing_scopes = {n.scope for n in node.out_nodes
                               if n.scope != node.scope}

            out_nodes = {node_of_scope[out_node]
                         for out_node in outgoing_scopes}
            in_nodes = {node_of_scope[in_node]
                        for in_node in incoming_scopes}

            node.in_nodes = in_nodes
            node.out_nodes = out_nodes

        self.adjacency_list = optimized_graph

    def _remove_constant_nodes(self):
        # remove nodes representing constants as they do not provide any useful info
        optimized_graph = []
        for node in self.adjacency_list:
            if "::Constant" in node.scope:
                for out_node in node.out_nodes:
                    out_node.remove_in_node(node)
                # just for sanity should never happen
                for in_node in node.in_nodes:
                    in_node.remove_out_node(node)
            else:
                optimized_graph.append(node)

        self.adjacency_list = optimized_graph

    def _merge_op_chains(self):
        # op chains need to be placed on the same device anyways
        optimized_graph = []
        for node in self.adjacency_list:
            # if OP flows only into other ops then remove it and connect it's inputs to it's outputs
            if node.type == NodeTypes.OP and len(node.out_nodes) > 0 and (o.type == NodeTypes.OP for o in node.out_nodes):
                for in_node in node.in_nodes:
                    in_node.remove_out_node(node)
                    in_node.add_out_node(node.out_nodes)
                for out_node in node.out_nodes:
                    out_node.remove_in_node(node)
                    out_node.add_in_node(node.in_nodes)
            else:
                optimized_graph.append(node)

        self.adjacency_list = optimized_graph

    def __getitem__(self, key):
        return self.adjacency_list[key]

    def partition(self, num_parts):
        return


# return a trace graph of a model convenience method
def trace_graph(model, *sample_inputs, optimized=True, op_type=torch.onnx.OperatorExportTypes.RAW):
    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(model, sample_inputs)
        # if optimized:
        #     trace_graph.set_graph(_optimize_graph(
        #         trace_graph.graph(), op_type))

        trace_graph = trace_graph.graph()

        return trace_graph


# optimizes a graph using gives op type
# a copy of torch.onnx.utils._optimize_graph
def _optimize_graph(graph, operator_export_type):
    # TODO there is a bug with the second scope name of sequential is carried to all layers after it in the sequence
    # maybe can be fixed bug is in torch/onnx/utils.py/139
    # TODO acctualy it appears we perform sufficient optimizations on our own
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
