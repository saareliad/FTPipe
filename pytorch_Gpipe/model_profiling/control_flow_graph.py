from copy import copy
from enum import Enum
from typing import Any, Dict, List


class Graph():
    '''
    a Graph data structure that model a pytorch network built from a pytorch trace\n
    the nodes operations like layer,Tensor ops etc.
    the edges represent the data flow in the model.
    names of the nodes are the scope names of their respective operations in the model.
    the graph can have weighted nodes
    do not instanciate this class directly use the graph_builder method provided with this module
    '''
    #TODO tuple inputs do not count as one graph cant distinguish between a tuple input/output and multiple input/outputs
    def __init__(self, profiled_layers: List[str], num_inputs: int, buffer_param_names: List[str], trace_graph, weights: Dict[str, Any],basic_blocks:List,depth:int):
        self.nodes = []
        self.profiled_layers = profiled_layers
        self.num_inputs_buffs_params = 0
        #TODO tuple inputs do not count as one
        self.num_inputs = num_inputs
        self.buffer_param_names = buffer_param_names
        self._build_graph(trace_graph)
        self.basic_blocks=basic_blocks
        self.depth=depth

        for node in self.nodes:
            node.weight = weights.get(node.scope, node.weight)

    def _build_graph(self, trace_graph):
        self._add_IO_nodes(trace_graph.inputs())
        self._add_OP_nodes(trace_graph.nodes())
        self._add_shapes(trace_graph)
        self._remove_constant_nodes()
        self._remove_nodes_that_go_nowhere(trace_graph.outputs())
        for idx,node in enumerate(self.nodes):
            node.idx=idx

    def _add_IO_nodes(self, input_nodes):
        '''
        add nodes representing the input and params/buffs of the model
        '''
        for idx, trace_node in enumerate(input_nodes):
            node_weight = 1
            # input/buff/parm weight is it's size
            for d in trace_node.type().sizes():
                node_weight *= d

            if idx < self.num_inputs:
                node_type = NodeTypes.IN
                node_scope = f"input{idx}"
            else:
                node_type = NodeTypes.BUFF_PARAM
                node_scope = self.buffer_param_names[idx - self.num_inputs]

            new_node = Node(node_scope, idx, node_type, weight=node_weight)
            self.nodes.append(new_node)

            self.num_inputs_buffs_params += 1

    def _add_OP_nodes(self, OP_nodes):
        '''
        add nodes representing the layers/ops of the model
        '''
        num_extra_nodes = 0
        for idx, trace_node in enumerate(OP_nodes):
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            input_nodes = {self.nodes[i.unique()]
                           for i in trace_node.inputs()}
            node_idx = self.num_inputs_buffs_params+idx+num_extra_nodes

            new_node = None

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

            self.nodes.append(new_node)

            # add node for each output
            for i, _ in enumerate(trace_node.outputs()):
                if i != 0:
                    out_node: Node = copy(new_node)
                    # it appears those are dummpy outputs so we just add them for bookkeeping and remove them later
                    out_node.out_nodes = set()
                    out_node.idx += i
                    self.nodes.append(out_node)
                    num_extra_nodes += 1

    def _add_shapes(self, trace_graph):
        '''
        add the shapes of all intermediate outputs and inputs to the graph nodes
        '''
        def get_shape(n):
            try:
                # works if not constant
                shape = tuple(n.type().sizes())

                if len(shape) == 0:
                    shape = (1,)

            except RuntimeError as _:
                # crashes for constant
                shape = (1,)
            return tuple(shape,)

        idx = 0
        output_idx = 0
        for node in trace_graph.inputs():
            u = self.nodes[idx]
            layer_out = LayerOutput(output_idx, u.scope, get_shape(node))
            u.outputs.add(layer_out)
            out_scopes=set()

            for use in node.uses():
                target_node = use.user
                # find the node idx of the user
                for out in target_node.outputs():
                    v = self.nodes[out.unique()]
                    v.inputs.add(layer_out)
                    out_scopes.add(v.scope)
            layer_out.out_scopes=out_scopes

            output_idx += 1
            idx += 1

        for node in trace_graph.nodes():
            for out in node.outputs():
                u = self.nodes[idx]
                layer_out = LayerOutput(output_idx, u.scope, get_shape(out))
                u.outputs.add(layer_out)
                out_scopes=set()
                for use in out.uses():
                    target_node = use.user
                    for target_out in target_node.outputs():
                        v = self.nodes[target_out.unique()]
                        v.inputs.add(layer_out)
                        out_scopes.add(v.scope)

                layer_out.out_scopes=out_scopes
                idx += 1
                output_idx += 1

    def _remove_constant_nodes(self):
        ''' remove nodes representing constants as they do not provide any useful info'''
        self._remove_nodes(lambda n: "::Constant" in n.scope)

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

    def _remove_nodes_that_go_nowhere(self, trace_outputs):
        '''remove nodes without out edges that are not outputs of the model'''
        #necessary because the trace can contain such nodes for certain ops
        #those nodes provide no additional info to the graph
        out_list = list(trace_outputs)
        out_indices = list(map(lambda n: int(n.uniqueName()),out_list))

        def going_nowhere(node):
            return (not node.out_nodes) and (not node.idx in out_indices)

        self._remove_nodes(going_nowhere)

    def _remove_nodes(self, condition, reverse:bool=False):
        changed = True
        while changed:
            changed = False
            optimized_graph = []

            nodes = reversed(self.nodes) if reverse else self.nodes

            for node in nodes:
                if condition(node):
                    changed = True
                    # connect inputs to outputs directly
                    for in_node in node.in_nodes:
                        in_node.remove_out_node(node)
                        in_node.add_out_node(node.out_nodes)
                    for out_node in node.out_nodes:
                        out_node.remove_in_node(node)
                        out_node.inputs.difference_update(node.outputs)
                        out_node.add_in_node(node.in_nodes)
                        out_node.inputs.update(node.inputs)
                else:
                    optimized_graph.append(node)

            self.nodes = optimized_graph

    def __getitem__(self, key):
        if isinstance(key,int):
            return self.nodes[key]
        # assume key is scopeName
        for node in self.nodes:
            if node.scope == key:
                return node
        return None

    def scopes(self)->List[str]:
        return list(map(lambda n:n.scope,self.nodes))

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        discription = ''
        for node in self.nodes:
            discription = f"{discription}\n{node}"
        return discription

    def get_nodes(self):
        return self.nodes

    def get_weights(self)->Dict[str,Any]:
        return {node.scope:node.weight for node in self.nodes}

    def adjacency_list(self, directed=False)->List[List[int]]:
        '''
        returns an adjacency list of the graph
        Parameters
        ----------
        directed:
            whether the adjacency list will be of the directed graph or the undirected graph 
        '''
        if not directed:
            return [[n.idx for n in node.out_nodes.union(node.in_nodes)] for node in self.nodes]
        return [[n.idx for n in node.out_nodes] for node in self.nodes]


    def build_dot(self, show_buffs_params=False, show_weights=True):
        '''
        return a graphviz representation of the graph
        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_weights:
            whether to display the nodes weight
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
                 rankdir="TB",
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

        #TODO split big graphs to multiple pdfs

        colors = {-1:'grey',0:'grey',1:'green',2:'red',3:'yellow',4:'orange',5:'brown',6:'purple',7:'pink'}

        def hide_node(node):
            return (node.type == NodeTypes.BUFF_PARAM) and (not show_buffs_params)

        for node in self.nodes:
            if hide_node(node):
                continue
            label = node.scope

            if not node.out_nodes:
                outputs = list(map(str, node.outputs))
                outputs = ",".join(outputs)
                label=f"{label}\n {outputs}"

            if show_weights and node.weight != 0:
                label = f"{label}\n {node.weight}"
            dot.node(str(node.idx), label, fillcolor=colors[node.part])

        for node in self.nodes:
            if hide_node(node):
                continue
            for in_node in node.in_nodes:
                if hide_node(in_node):
                    continue
                edge_label = filter(lambda layer_in: layer_in.scope ==
                               in_node.scope, node.inputs)

                edge_label=list(map(str,edge_label))
                edge_label=",".join(edge_label)
                dot.edge(str(in_node.idx), str(node.idx), label=edge_label)

        return dot

    def display(self, show_buffs_params=False, show_weights=True):
        '''
        display the graph in Jupyter

        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_weights:
            whether to display the nodes weight
        '''
        try:
            from IPython.core.display import display_svg
            display_svg(self.build_dot(show_buffs_params, show_weights=show_weights), raw=False)
        except ImportError as _:
            print("only works in python notebooks")

    def save(self, file_name,directory, show_buffs_params=False,show_weights=True):
        '''
        save the graph to a file

        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_weights:
            whether to display the nodes weight
        '''
        dot = self.build_dot(show_buffs_params, show_weights=show_weights)
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)


class NodeTypes(Enum):
    '''
    Enum representing the possible types of Nodes in the Graph
    '''
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4

    def __repr__(self):
        return self.name


class Node():
    '''
    a simple graph node for weighted directed graphs

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
    inputs:
        the LayerOutputs that consumed by this Node
    outputs:
        the LayerOutputs produced by this Node
    weight:
        the weight of the edge can be anything
    part:
        partition idx determines the color of the Node
    
     parallel edges in the same direction are not allowed
    '''

    def __init__(self, scope:str, idx:int, node_type: NodeTypes, incoming_nodes=None, weight=0, part=-1):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes = set()
        self.weight = weight
        self.part = part
        self.in_nodes = incoming_nodes if isinstance(
            incoming_nodes, set) else set()
        self.outputs = set()
        self.inputs = set()

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


class LayerOutput():
    '''
    a simple class representing a layer output

    Fields
    ----------
    idx:
        a unique index of this output
    origin_scope:
        the scope which produces this output
    output_shape:
        the shape of this output
    '''
    def __init__(self, idx, origin_scope, output_shape):
        self.idx = idx
        self.scope = origin_scope
        self.output_shape = output_shape
        self.out_scopes = set()

    def __eq__(self, other):
        if isinstance(other,tuple):
            return other == tuple(self.output_shape)

        if not isinstance(other, LayerOutput):
            return False

        return self.idx == other.idx

    def __hash__(self):
        return self.idx.__hash__()

    def __str__(self):
        res = ''
        for d in self.output_shape:
            res = f"{res}x{d}"
        return res[1:]

    def __repr__(self):
        return str(self)
