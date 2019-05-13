
import torch.nn as nn
import torch
from enum import Enum
from pprint import pprint

__all__ = ['build_control_flow_graph']


def build_control_flow_graph(model, *sample_batch, max_depth=100, weights=None, basic_block=None, device="cuda"):
    model_class_name = type(model).__name__
    
    buffer_names = _profiled_params_buffs(
        model, max_depth, prefix=model_class_name,generator=nn.Module.named_buffers, basic_block=basic_block)
    
    parameter_names = _profiled_params_buffs(
        model, max_depth, prefix=model_class_name, generator=nn.Module.named_parameters, basic_block=basic_block)

    weights = weights if weights != None else {}

    layerNames = _profiled_layers(
        model, max_depth, prefix=model_class_name, basic_block=basic_block)

    # trace the model and build a graph
    inputs = tuple(map(lambda t: t.to(device), sample_batch))
    model.to(device)
    num_inputs = len(inputs)

    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(model, inputs)
        trace_graph = trace_graph.graph()

    return Graph(layerNames, num_inputs, buffer_names, trace_graph, weights)


class NodeTypes(Enum):
    IN = 1
    BUFF = 2
    PARAM = 3
    LAYER = 4
    OP = 5

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

    def __init__(self, scope, idx, node_type: NodeTypes, incoming_nodes=None, weight=0, part=0):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes = set()
        self.weight = weight
        self.part = part
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

    def __init__(self, profiled_layers, num_inputs, buffer_names, trace_graph, weights: dict):
        self.nodes = []
        self.profiled_layers = profiled_layers
        self.num_inputs_buffs_params = 0
        self.num_inputs = num_inputs
        self.buffer_names = buffer_names
        self._build_graph(trace_graph)

        for node in self.nodes:
            node.weight = weights.get(node.scope, node.weight)

    def _build_graph(self, trace_graph):
        self._add_IO_nodes(trace_graph.inputs())
        self._add_OP_nodes(trace_graph.nodes())
        self._combine_nodes_under_the_same_scope()
        self._remove_constant_nodes()
        self._merge_op_chains()
        self._normalize_indices()

    def _add_IO_nodes(self, input_nodes):
        '''
        add nodes representing the input and params/buffs of the model
        '''
        for idx, node in enumerate(input_nodes):
            # TODO what if buffer is not used? remove or add
            # if len(list(node.uses())) == 0:
            #     continue
            node_weight = 1
            # input/buff/parm weight is it's size
            # TODO normalize
            for d in node.type().sizes():
                node_weight *= d

            if idx < self.num_inputs:
                node_type = NodeTypes.IN
                node_scope = f"input{idx}"
            elif idx < self.num_inputs + len(self.buffer_names):
                node_type = NodeTypes.BUFF
                node_scope = self.buffer_names[idx-self.num_inputs]
            else:
                node_type = NodeTypes.PARAM
                node_scope = f"param{idx-self.num_inputs-len(self.buffer_names)}"

            new_node = Node(node_scope, idx,
                            node_type, weight=node_weight)
            self.nodes.append(new_node)

            self.num_inputs_buffs_params += 1

    def _add_OP_nodes(self, OP_nodes):
        '''
        add nodes representing the layers/ops of the model
        '''
        for idx, trace_node in enumerate(OP_nodes):
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            input_nodes = {self.nodes[i.unique()]
                           for i in trace_node.inputs()}

            node_idx = self.num_inputs_buffs_params+idx
            new_node = None

            # profiled Layer
            if node_scope != "":
                # TODO what about function modules like ReLU which are used multiple times
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

            self.nodes.append(new_node)

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
        for node in self.nodes:
            if not node.scope in node_of_scope:
                optimized_graph.append(node)
                node_of_scope[node.scope] = node

            else:
                # add edges create the super set of all edeges in the scope
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

        self.nodes = optimized_graph

    def _remove_constant_nodes(self):
        # remove nodes representing constants as they do not provide any useful info
        self._remove_nodes(lambda n: "::Constant" in n.scope)

    def _merge_op_chains(self):
        def to_remove(n): return n.type == NodeTypes.OP and len(n.out_nodes) > 0 and all(
            o.type == NodeTypes.OP for o in n.out_nodes)
        # op chains need to be placed on the same device anyways
        self._remove_nodes(to_remove)

    def _remove_nodes(self, condition):
        optimized_graph = []
        for node in self.nodes:
            if condition(node):
                # connect inputs to outputs directly
                for in_node in node.in_nodes:
                    in_node.remove_out_node(node)
                    in_node.add_out_node(node.out_nodes)
                for out_node in node.out_nodes:
                    out_node.remove_in_node(node)
                    out_node.add_in_node(node.in_nodes)
            else:
                optimized_graph.append(node)

        self.nodes = optimized_graph

    def __getitem__(self, key):
        return self.nodes[key]

    def __repr__(self):
        discription = ''
        for node in self.nodes:
            discription = f"{discription}\n{node}"
        return discription

    def get_nodes(self):
        return self.nodes

    def get_weights(self):
        return [node.weight for node in self.nodes]

    def adjacency_list(self,directed=False):
        if not directed:
            return [[n.idx for n in node.out_nodes.union(node.in_nodes)] for node in self.nodes]
        return [[n.idx for n in node.out_nodes] for node in self.nodes]

    def _normalize_indices(self):
        for idx, node in enumerate(self.nodes):
            node.idx = idx

    def set_partition(self, parts):
        for node, part in zip(self.nodes, parts):
            node.part = part

    def build_dot(self, show_buffs=False, show_params=False):
        '''
        return a graphviz representation of the graph
        '''

        theme = {"background_color": "#FFFFFF",
                 "fill_color": "#E8E8E8",
                 "outline_color": "#000000",
                 "font_color": "#000000",
                 "font_name": "Times",
                 "font_size": "10",
                 "margin": "0,0",
                 "padding":  "1.0,0.5"}
        from graphviz import Digraph

        dot = Digraph()

        dot.attr("graph",
                 concentrate="true",
                 bgcolor=theme["background_color"],
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"],
                 margin=theme["margin"],
                 rankdir="LR",
                 pad=theme["padding"])

        dot.attr("node", shape="box",
                 style="filled", margin="0,0",
                 fillcolor=theme["fill_color"],
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"])

        dot.attr("edge", style="solid",
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"])

        colors={0:'blue',1:'green',2:'red',3:'yellow',4:'orange',5:'brown',6:'purple',7:'pink'}


        def hide_node(node):
            return(node.type == NodeTypes.BUFF and not show_buffs) or (node.type == NodeTypes.PARAM and not show_params)

        for node in self.nodes:
            if hide_node(node):
                continue
            label = node.scope
            if node.weight != 0:
                label = f"{label}\n {node.weight}"
            dot.node(str(node.idx), label, fillcolor=colors[node.part])

        for node in self.nodes:
            if hide_node(node):
                continue
            for in_node in node.in_nodes:
                if hide_node(in_node):
                    continue
                dot.edge(str(in_node.idx), str(node.idx))
        return dot

    def display(self, show_buffs=False, show_params=False):
        try:
            from IPython.core.display import display_svg
            display_svg(self.build_dot(show_buffs=show_buffs,
                                       show_params=show_params), raw=False)
        except ImportError as e:
            print("only works in python notebooks")
            pass


# scope names of all profiled layers in the model
def _profiled_layers(module: nn.Module, depth, prefix, basic_block):
    names = []
    for name, sub_module in module._modules.items():
        # assume no cyclic routes in the network
        # a module with no children is a layer
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            names.append(
                prefix+"/"+type(sub_module).__name__+f"[{name}]")
        else:
            names = names + _profiled_layers(sub_module, depth-1, prefix +
                                             "/"+type(sub_module).__name__+f"[{name}]",basic_block)
    return names

#TODO return buffer names/params in desired format
# scope names of all profiled params and buffs in the model
def _profiled_params_buffs(module: nn.Module, depth, prefix,generator, basic_block):
    names = []
    for name, sub_module in module._modules.items():
        # assume no cyclic routes in the network
        # a module with no children is a layer
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            prefix = prefix+"/"+type(sub_module).__name__+f"[{name}]"
            for item_name,_ in sub_module.named_buffers():
                names.append(f"{prefix}/{item_name}")
        else:
            names = names + _profiled_params_buffs(sub_module, depth-1, prefix +
                                             "/"+type(sub_module).__name__+f"[{name}]",generator,basic_block)
    return names
