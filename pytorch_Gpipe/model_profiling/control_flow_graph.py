from enum import Enum
from typing import Any, Dict, List
from ..utils import OrderedSet
import string
import inspect
import torch
import re

class Graph():
    '''
    a Graph data structure that model a pytorch network built from a pytorch trace\n
    the nodes operations like layer,Tensor ops etc.
    the edges represent the data flow in the model.
    names of the nodes are the scope names of their respective operations in the model.
    the graph can have weighted nodes
    do not instanciate this class directly use the graph_builder method provided with this module
    '''

    def __init__(self, profiled_layers: List[str], num_inputs: int, buffer_param_names: List[str], trace_graph, weights: Dict[str, Any], basic_blocks: List, depth: int):
        self.nodes = []
        self.profiled_layers = profiled_layers
        self.num_inputs_buffs_params = 0
        self.num_inputs = num_inputs
        self.buffer_param_names = buffer_param_names
        self.model_name = profiled_layers[0].split('/')[0]
        self._build_graph(trace_graph)
        self.basic_blocks = basic_blocks
        self.depth = depth
        self.num_parts = 0
        
        for node in self.nodes:
            node.weight = weights.get(node.scope, node.weight)

    def _build_graph(self, trace_graph):
        self._add_IO_nodes(trace_graph.inputs())
        self._add_OP_nodes(trace_graph.nodes())
        #TODO we've disabled output shape untill we can think about full support
        # self._add_shapes(trace_graph)
        self._set_outputs(trace_graph.outputs())
        
        self.remove_useless_clone()   
        self.remove_empty_view()
        optimize_graph(self)
        self._remove_nodes_that_go_nowhere(trace_graph)
        for idx, node in enumerate(self.nodes):
            node.idx = idx
        self.remove_useless_node_inputs()
        self.add_missing_types()
        self.remove_tensor_int_tensor()
        for idx, node in enumerate(self.nodes):
            node.idx = idx

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
            new_node.value_type=torch.Tensor
            self.nodes.append(new_node)

            self.num_inputs_buffs_params += 1

    def _add_OP_nodes(self, OP_nodes):
        '''
        add nodes representing the layers/ops of the model
        '''
        num_extra_nodes = 0
        for idx, trace_node in enumerate(sorted(OP_nodes, key=lambda n: next(n.outputs()).unique())):
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            input_nodes = OrderedSet([self.nodes[i.unique()]
                           for i in trace_node.inputs()])
            node_idx = self.num_inputs_buffs_params + idx + num_extra_nodes
            new_node = None

            # profiled Layer
            if node_scope != "":
                new_node = Node(node_scope, node_idx,
                                NodeTypes.LAYER, input_nodes)
                new_node.value_type=torch.Tensor
            # unprofiled constant value
            elif 'prim::Constant' in trace_node.kind():
                node_scope = trace_node.scopeName() + \
                    "/" + trace_node.kind() + str(node_idx - self.num_inputs_buffs_params)
                value = trace_node.output().toIValue()
                new_node = Node(node_scope, node_idx,
                                NodeTypes.CONSTANT, input_nodes, value=value)
            else:
                # unprofiled List
                if 'prim::' in trace_node.kind():
                    node_type = NodeTypes.PYTHON_PRIMITIVE
                # unprofiled torch op
                # TODO should we specialize the aten:: and prim:: cases
                elif 'aten::' in trace_node.kind():
                    node_type = NodeTypes.OP
                else:
                    # unprofiled other
                    assert False, f"unknown scope {trace_node.scopeName()}"

                node_scope = trace_node.scopeName() + \
                    "/" + trace_node.kind() + str(node_idx - self.num_inputs_buffs_params)
                new_node = Node(node_scope, node_idx,
                                node_type, input_nodes)

            # add incoming edges
            for node in input_nodes:
                node.add_out_node(new_node)

            self.nodes.append(new_node)

            nOuts = 1
            # add node for each output
            for i, output in enumerate(trace_node.outputs()):
                if i==0 and output.isCompleteTensor():
                    self.nodes[-1].value_type=torch.Tensor
                if i != 0:
                    out_scope = new_node.scope
                    if self._find_encasing_layer(new_node.scope) == "":
                        out_scope += f"{i} "
                    out_idx = new_node.idx+i
                    out_node = Node(out_scope,out_idx,new_node.type)
                    out_node.add_in_node(new_node.in_nodes[0])
                    new_node.in_nodes[0].add_out_node(out_node)
                    if output.isCompleteTensor():
                        out_node.value_type=torch.Tensor
                    self.nodes.append(out_node)
                    num_extra_nodes += 1
                    nOuts+=1
            if nOuts > 1 and self._find_encasing_layer(self.nodes[-nOuts].scope) == "":
                self.nodes[-nOuts].scope+=f"{0} "

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
            out_scopes=OrderedSet()

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
                out_scopes=OrderedSet()
                for use in out.uses():
                    target_node = use.user
                    for target_out in target_node.outputs():
                        v = self.nodes[target_out.unique()]
                        v.inputs.add(layer_out)
                        out_scopes.add(v.scope)

                layer_out.out_scopes=out_scopes
                idx += 1
                output_idx += 1

    def add_missing_types(self):
        for node in self.nodes:
            if node.valueType() is type(None):
                if 'aten::size' in node.scope or 'aten::Int' in node.scope:
                    node.value_type=int
                elif 'prim::ListConstruct' in node.scope or 'aten::chunk' in node.scope:
                    node.value_type=list
                elif 'ImplicitTensorToNum' in node.scope:
                    node.value_type=int
            elif 'NumToTensor' in node.scope:
                node.value_type=int

    def remove_tensor_int_tensor(self):
        def predicate(node):
            if 'prim::ImplicitTensorToNum' in node.scope or 'aten::Int' in node.scope or 'aten::NumToTensor' in node.scope:
                for n in node.in_nodes:
                    n.value_type = int
                return True
            return False

        self._remove_nodes(predicate)


    def remove_useless_clone(self):
        def predicate(n:Node):
            return ('aten::clone' in n.scope) and (len(n.out_nodes) == 0)
        self._remove_nodes(predicate)

    def remove_empty_view(self):
        def predicate(n:Node):
            if ('aten::view' in n.scope):
                if len(n.in_nodes) < 2:
                    return True
                sizes = list(n.in_nodes)[1]
                return len(sizes.in_nodes) == 0
            return('prim::ListConstruct' in n.scope) and (len(n.in_nodes) == 0)
        self._remove_nodes(predicate)

    def remove_useless_node_inputs(self):
        # stupid fix where for some odd reason arithmetic ops have a third input with value 1
        # and Tensor.contiguous has a second input with value 0
        # and torch.arange having a zero input
        def pred(node:Node):
            if node.type == NodeTypes.CONSTANT and (node.value in [0,1]):
                assert len(node.out_nodes) == 1 , "Constant should have one use"
                out = node.out_nodes[0]
                arithmetic_ops = ['aten::add','aten::div','aten::mul','aten::sub']
                arithmetic = any(opMatch(out.scope,o) for o in arithmetic_ops) and (out.in_nodes.indexOf(node) == 2)
                contiguous_input = ('aten::contiguous' in out.scope) and (
                    out.in_nodes.indexOf(node) == 1)
                arange_input = ('aten::arange' in out.scope) and (
                    out.in_nodes.indexOf(node) == (len(out.in_nodes) - 3))
                return arithmetic or contiguous_input or arange_input
            return False
        self._remove_nodes(pred) 

    def _find_encasing_layer(self, scopeName: str):
        '''
        find the closest scope which encases scopeName
        '''
        # unfortunately the trace graph shows only basic layers and ops
        # so we need to manually find a profiled layer that encases the op
        most_specific_scope = ""
        for layer_scope in self.profiled_layers:
            if scopeName.startswith(layer_scope):
                most_specific_scope = layer_scope
                break
        return most_specific_scope

    def _remove_nodes_that_go_nowhere(self, trace_graph):
        '''remove nodes without out edges that are not outputs of the model'''
        # necessary because the trace can contain such nodes for certain ops
        # those nodes provide no additional info to the graph    
        out_indices=[self._get_id(out) for out in trace_graph.outputs()]
        
        def going_nowhere(node):
            if node.type is NodeTypes.OP and 'aten::' in node.scope:
                func_name = node.scope.split('aten::')[1].rstrip(string.digits)
                # do not remove inplace ops prematurly
                if func_name[-1] == '_':
                    return False

            if node.scope in self.output_scopes:
                return False
                    
            return (not node.out_nodes) and (not node.idx in out_indices)

        self._remove_nodes(going_nowhere)

    def _get_id(self,out):
        # we need this method for compatibility issues
        # in pytorch 1.2.0 the API changed the method name from uniqueName to debugName
        # maybe it's a sign that we should not relay on it but it's simple and effective...
        if hasattr(out, 'debugName'):
            # 1.2.0 and onward
            n = out.debugName()

        else:
            # before 1.2.0
            assert hasattr(out, 'uniqueName')
            n = out.uniqueName()
        return int(n)

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
                    # TODO we do not remove/add inputs or outputs might revisit
                    for in_node in node.in_nodes:
                        in_node.replace_out_node(node,node.out_nodes)
                        if node.value_type:
                            in_node.value_type=node.value_type
                            in_node.value=None
                    for out_node in node.out_nodes:
                        out_node.replace_in_node(node,node.in_nodes)
                        out_node.inputs.difference_update(node.outputs)
                        out_node.inputs.update(node.inputs)
                else:
                    optimized_graph.append(node)

            self.nodes = optimized_graph

    def _set_outputs(self,trace_outputs):
        outputs=OrderedSet()
        for out in trace_outputs:
            node=out.node()
            scope=self._find_encasing_layer(node.scopeName())
            if scope == '':
                idx = self._get_id(out) - self.num_inputs_buffs_params
                scope=node.scopeName() + \
                    "/" + node.kind() + str(idx)
            outputs.add(scope)
        self.output_scopes=outputs

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

    def asNetworkx(self):
        try:
            import networkx as nx
        except ImportError as _:
            print("networkx package not found")
            return
        
        #edge_list
        edge_list=[]
        for u in self.nodes:
            for v in u.in_nodes:
                edge_list.append((u.idx,v.idx))

        G = nx.from_edgelist(edge_list)
        for n in self.nodes:
            G.nodes[n.idx]['weight']=n.weight
        
        return G

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

        # TODO split big graphs to multiple pdfs

        colors = {0:'grey',1:'green',2:'red',3:'yellow',4:'orange',5:'brown',6:'purple',7:'pink'}

        def hide_node(node):
            return (node.type == NodeTypes.BUFF_PARAM) and (not show_buffs_params)

        for node in self.nodes:
            if hide_node(node):
                continue
            label = f"{node.scope} {node.idx}"

            if not node.out_nodes:
                outputs = list(map(str, node.outputs))
                outputs = ",".join(outputs)
                label=f"{label}\n {outputs}"

            if show_weights and node.weight != 0:
                label = f"{label}\n {node.weight}"

            label = f"{label}\n type: {node.valueType()}"
            if not (node.value is None):
                label = f"{label}\n value={node.value}"
            
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
        save the rendered graph to a file

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

    def serialize(self, file_name):
        import sys
        import pickle
        rec=sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        pickle.dump(self, open(file_name, "wb"))
        sys.setrecursionlimit(rec)

class NodeTypes(Enum):
    '''
    Enum representing the possible types of Nodes in the Graph
    '''
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4
    CONSTANT=5
    PYTHON_PRIMITIVE=6

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

    def __init__(self, scope:str, idx:int, node_type: NodeTypes, incoming_nodes=None, weight=0, part=0,value=None):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes = OrderedSet()
        self.weight = weight
        self.part = part
        self.in_nodes = incoming_nodes if isinstance(
            incoming_nodes, OrderedSet) else OrderedSet()
        self.outputs = OrderedSet()
        self.inputs = OrderedSet()
        self.value=value
        self.value_type=None

    def valueType(self):
        if self.value_type:
            return self.value_type
        else:
            return type(self.value)

    def add_out_node(self, node):
        if isinstance(node, Node):
            self.out_nodes.add(node)
        if isinstance(node, (set, OrderedSet)):
            self.out_nodes.update(node)

    def add_in_node(self, node):
        if isinstance(node, Node):
            self.in_nodes.add(node)
        if isinstance(node, (set, OrderedSet)):
            self.in_nodes.update(node)

    def remove_in_node(self, node):
        if isinstance(node, Node):
            self.in_nodes.discard(node)
        if isinstance(node, (set, OrderedSet)):
            self.in_nodes.difference_update(node)

    def remove_out_node(self, node):
        if isinstance(node, Node):
            self.out_nodes.discard(node)
        if isinstance(node, (set,OrderedSet)):
            self.out_nodes.difference_update(node)

    def replace_out_node(self,to_replace,value):
        if to_replace not in self.out_nodes:
            return
        
        values = list(self.out_nodes)
        idx = values.index(to_replace)

        before,after=values[:idx],values[idx+1:]
        try:
            # we handle the case for iterable, if value is not then we recall with [value]
            iter(value)
            keys=value
            to_add = [v for v in keys if (v not in before) and (v not in after)]
            self.out_nodes=OrderedSet(before+to_add+after)

        except TypeError as _:
            self.replace_out_node(to_replace,[value])

    def replace_in_node(self,to_replace,value):
        if to_replace not in self.in_nodes:
            return

        values = list(self.in_nodes)
        idx = values.index(to_replace)

        before, after = values[:idx], values[idx + 1:]
        try:
            # we handle the case for iterable, if value is not then we recall with [value]
            iter(value)
            keys=value
            to_add = [v for v in keys if (v not in before) and (v not in after)]
            self.in_nodes = OrderedSet(before + to_add + after)

        except TypeError as _:
            self.replace_in_node(to_replace, [value])

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


def optimize_graph(graph: Graph):
    '''
    this module takes the raw Graph and removes/merges nodes in order to get the requested graph.
    this method is called as part of graph_builder method
    '''
    nodes = graph.nodes
    nodes = _combine_OP_nodes_under_the_same_scope(nodes)
    graph.nodes = nodes
    _combine_params_and_buffers_into_OP_nodes(graph)
    # TODO we disabled op chain merges
    # _merge_op_chains(graph)


def _combine_OP_nodes_under_the_same_scope(nodes: List[Node]) -> List[Node]:
    # optimization that reduces number of nodes in the graph
    # combine nodes that have a commom scope we do this because\n
    # if nodes have the same scopeName than they were profiled together
    scope_representative = dict()

    optimized_graph = []

    # get the nodes of the optimized graph
    for node in nodes:
        if not node.scope in scope_representative:
            optimized_graph.append(node)
            scope_representative[node.scope] = node
        else:
            # add edges create the super set of all edeges in the scope
            scope_representative[node.scope].add_in_node(node.in_nodes)
            scope_representative[node.scope].inputs.update(node.inputs)

            scope_representative[node.scope].add_out_node(node.out_nodes)
            scope_representative[node.scope].outputs.update(node.outputs)

    for node in optimized_graph:
        # get the sets of all incoming/outgoing scopes
        # those will dictate the new set of edges and
        # remove the internal edges of the scope
        incoming_scopes = OrderedSet(n.scope for n in node.in_nodes
                                     if n.scope != node.scope)
        outgoing_scopes = OrderedSet(n.scope for n in node.out_nodes
                                     if n.scope != node.scope)

        inputs = OrderedSet(layer_in for layer_in in node.inputs
                            if layer_in.scope != node.scope)
        outputs = {layer_out for layer_out in node.outputs
                   if node.scope not in layer_out.out_scopes}

        out_nodes = OrderedSet(scope_representative[out_node]
                               for out_node in outgoing_scopes)
        in_nodes = OrderedSet(scope_representative[in_node]
                              for in_node in incoming_scopes)

        node.in_nodes = in_nodes
        node.out_nodes = out_nodes
        node.inputs = inputs
        node.outputs = outputs

    return optimized_graph


def _combine_params_and_buffers_into_OP_nodes(graph: Graph):
    def is_buffer_or_param(n):
        return n.type == NodeTypes.BUFF_PARAM and graph._find_encasing_layer(n.scope) != ''

    graph._remove_nodes(is_buffer_or_param)


def _merge_op_chains(graph: Graph):
    def to_remove(n): return n.type == NodeTypes.OP and len(n.out_nodes) > 0 and all(
        o.type == NodeTypes.OP for o in n.out_nodes)

    # op chains need to be placed on the same device anyways
    graph._remove_nodes(to_remove)

def opMatch(scope,op_name):
    return re.search(f"{op_name}[{string.digits}]",scope)