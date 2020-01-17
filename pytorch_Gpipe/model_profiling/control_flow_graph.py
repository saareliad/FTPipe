from enum import Enum
from typing import Any, Dict, List, Tuple, Optional
from ..utils import OrderedSet
import string
import inspect
import torch
import torch.nn as nn
import re
from copy import deepcopy
from collections import OrderedDict

# TODO support list and tuple layer outputs


class OLDGraph():
    '''
    a Graph data structure that model a pytorch network built from a pytorch trace\n
    the nodes operations like layer,Tensor ops etc.
    the edges represent the data flow in the model.
    names of the nodes are the scope names of their respective operations in the model.
    the graph can have weighted nodes
    do not instanciate this class directly use the graph_builder method provided with this module
    '''

    def __init__(self, profiled_layers: List[str], num_inputs: int, tensors: List[Tuple[str, Tuple[int]]], trace_graph, weights: Dict[str, Any], depth: int, basic_blocks: Optional[List[nn.Module]] = None, use_jit_trace=False):
        # assert use_jit_trace, "_get_trace_graph is currently broken"
        self.nodes = OrderedDict()
        self.profiled_layers = translate_scopes(profiled_layers)
        self.partial_scopes = partial_scopes(profiled_layers)
        self.num_inputs = num_inputs
        self._build_graph(trace_graph, tensors, use_jit_trace)
        self.depth = depth
        self.basic_blocks = tuple() if basic_blocks is None else tuple(basic_blocks)

        normalized_nodes = OrderedDict()
        for idx, node in enumerate(self.nodes.values()):
            node.weight = weights.get(node.scope, node.weight)
            node.idx = idx
            normalized_nodes[idx] = node

        self.nodes = normalized_nodes

    def _build_graph(self, trace_graph, tensors, use_jit_trace):
        offset = self._add_IO_nodes(trace_graph, tensors, use_jit_trace)
        self._add_OP_nodes(list(trace_graph.nodes())[
                           offset - self.num_inputs:], use_jit_trace)
        # TODO we've disabled output shape untill we can think about full support
        # self._add_shapes(trace_graph)
        # self.save(f"verbose", f"playground_out/graphs/{self.model_name}")
        self._set_outputs(trace_graph.outputs(), use_jit_trace)
        self.remove_useless_clone()
        # self.save(f"remove_clones", f"playground_out/graphs/{self.model_name}")
        self.remove_empty_view()
        # self.save(f"remove_empty_views",
        #   f"playground_out/graphs/{self.model_name}")
        optimize_graph(self)
        # self.save(f"optimized", f"playground_out/graphs/{self.model_name}")
        self._remove_nodes_that_go_nowhere(trace_graph)
        # self.save(f"remove_goes_nowhere",
        #   f"playground_out/graphs/{self.model_name}")
        self.remove_useless_node_inputs()
        # self.save(f"removed_useless_inputs",
        #   f"playground_out/graphs/{self.model_name}")
        self.add_missing_types()
        self.remove_tensor_int_tensor()
        # self.save(f"remove_tensor_int_tensor",
        #   f"playground_out/graphs/{self.model_name}")

    def _add_IO_nodes(self, trace_graph, tensors, use_jit_trace):
        '''
        add nodes representing the input and params/buffs of the model
        '''
        num_inputs_buffs_params = 0
        if use_jit_trace:
            nodes = list(trace_graph.inputs())[1:] + list(trace_graph.nodes())
        else:
            nodes = trace_graph.inputs()

        for idx, trace_node in enumerate(nodes):
            # break condition for jit.trace because they do not include buffers/parameters as inputs
            if num_inputs_buffs_params == len(tensors):
                return idx
            # each graph starts with input and buff/param declarations and getattr nodes used to access the buff/param
            # we add only nodes whoes type is Tensor(buff/param) or nodes that represent inputs the first num_inputs nodes
            if (not use_jit_trace) or (idx < self.num_inputs or (trace_node.output().type().str() == "Tensor")):
                if num_inputs_buffs_params < self.num_inputs:
                    node_type = NodeTypes.IN
                    unique_id = trace_node.unique()
                else:
                    node_type = NodeTypes.BUFF_PARAM
                    unique_id = trace_node.output().unique() if use_jit_trace else trace_node.unique()

                node_scope = tensors[num_inputs_buffs_params][0]
                node_weight = 1
                # input/buff/parm weight is it's size
                for d in tensors[num_inputs_buffs_params][1]:
                    node_weight *= d

                new_node = Node(node_scope, unique_id, node_type,
                                weight=node_weight)
                new_node.value_type = torch.Tensor
                self.nodes[unique_id] = new_node
                num_inputs_buffs_params += 1

        return self.num_inputs

    def _add_OP_nodes(self, OP_nodes, use_jit_trace):
        '''
        add nodes representing the layers/ops of the model
        '''
        # TODO this is for 1.3.1
        # for trace_node in sorted(OP_nodes, key=lambda n: next(n.outputs()).unique()):
        # in 1.4 the output scope can is not largest id but he is last
        for trace_node in OP_nodes:
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            try:
                input_nodes = OrderedSet([self.nodes[i.unique()]
                                          for i in trace_node.inputs()])
            except Exception as e:
                print(trace_node)
                print(f"graph nodes are {self.nodes.keys()}")
                for i in trace_node.inputs():
                    print(i.unique())
                raise e

            new_node = None
            node_type = None
            value_type = None
            value = None
            if node_scope != "":
                # profiled Layer
                node_type = NodeTypes.LAYER
                value_type = torch.Tensor
            else:
                node_scope = self._find_partial_match(
                    trace_node.scopeName()) + "/" + trace_node.kind()
                if 'prim::Constant' in trace_node.kind():
                    # unprofiled constant value
                    node_type = NodeTypes.CONSTANT
                    value = trace_node.output().toIValue()
                elif 'prim::' in trace_node.kind():
                    # unprofiled List or Tuple
                    node_type = NodeTypes.PYTHON_PRIMITIVE
                elif 'aten::' in trace_node.kind():
                    # unprofiled torch op
                    # TODO should we specialize the aten:: and prim:: cases
                    node_type = NodeTypes.OP
                else:
                    # unprofiled other
                    assert False, f"unknown scope {trace_node.scopeName()}"

            if use_jit_trace and node_scope.startswith("/"):
                # TODO this is a bug in jit.trace it works fine with jit.get_trace
                # weird case where we don not have the model class as prefix
                print(
                    f"node without a legal scope\n got {trace_node.scopeName()}\n{trace_node}assuming a top level node")
                node_scope = self.model_name + node_scope

            nOuts = 1
            # add node for each output
            for i, output in enumerate(trace_node.outputs()):
                # TODO in some cases we can know the shape of the tensor per edge
                # try:
                #     print(output.type().sizes())
                # except Exception:
                #     print(node_scope)

                unique_id = output.unique()

                # to differentiate different non layer ops that are in the same scope
                if i == 0 and node_type != NodeTypes.LAYER:
                    node_scope += str(unique_id)
                # create new node
                new_node = Node(node_scope, unique_id,
                                node_type, input_nodes, value=value)

                # add incoming edges
                for node in input_nodes:
                    node.add_out_node(new_node)

                new_node.value_type = value_type
                self.nodes[unique_id] = new_node

                # if tensor node set type accordingly
                if output.isCompleteTensor():
                    new_node.value_type = torch.Tensor

                # secondery output
                if i != 0:
                    if self._find_encasing_layer(node_scope) == "":
                        new_node.scope += f"{i} "
                    new_node.add_in_node(self.nodes[unique_id - i].in_nodes[0])
                    self.nodes[unique_id -
                               i].in_nodes[0].add_out_node(new_node)
                    nOuts += 1

            if nOuts > 1 and self._find_encasing_layer(self.nodes[unique_id - i].scope) == "":
                self.nodes[unique_id - i].scope += "0 "

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
            out_scopes = OrderedSet()

            for use in node.uses():
                target_node = use.user
                # find the node idx of the user
                for out in target_node.outputs():
                    v = self.nodes[out.unique()]
                    v.inputs.add(layer_out)
                    out_scopes.add(v.scope)
            layer_out.out_scopes = out_scopes

            output_idx += 1
            idx += 1

        for node in trace_graph.nodes():
            for out in node.outputs():
                u = self.nodes[idx]
                layer_out = LayerOutput(output_idx, u.scope, get_shape(out))
                u.outputs.add(layer_out)
                out_scopes = OrderedSet()
                for use in out.uses():
                    target_node = use.user
                    for target_out in target_node.outputs():
                        v = self.nodes[target_out.unique()]
                        v.inputs.add(layer_out)
                        out_scopes.add(v.scope)

                layer_out.out_scopes = out_scopes
                idx += 1
                output_idx += 1

    def add_missing_types(self):
        for node in self.nodes.values():
            if node.valueType() is type(None):
                if 'aten::size' in node.scope or 'aten::Int' in node.scope:
                    node.value_type = int
                elif 'prim::ListConstruct' in node.scope or 'aten::chunk' in node.scope or 'prim::TupleConstruct':
                    node.value_type = list
                elif 'ImplicitTensorToNum' in node.scope:
                    node.value_type = int
            elif 'NumToTensor' in node.scope:
                node.value_type = int

    def remove_tensor_int_tensor(self):
        def predicate(node):
            if 'prim::ImplicitTensorToNum' in node.scope or 'aten::Int' in node.scope or 'prim::NumToTensor' in node.scope:
                for n in node.in_nodes:
                    n.value_type = int
                return True
            return False

        self._remove_nodes(predicate)

    def remove_useless_clone(self):
        def predicate(n: Node):
            return ('aten::clone' in n.scope) and (len(n.out_nodes) == 0)
        self._remove_nodes(predicate)

    def remove_empty_view(self):
        def predicate(n: Node):
            if ('aten::view' in n.scope):
                if len(n.in_nodes) < 2:
                    return True
                sizes = list(n.in_nodes)[1]
                return len(sizes.in_nodes) == 0
            return('prim::ListConstruct' in n.scope or 'prim::TupleConstruct' in n.scope) and (len(n.in_nodes) == 0)
        self._remove_nodes(predicate)

    def remove_useless_node_inputs(self):
        # stupid fix where for some odd reason arithmetic ops have a third input with value 1
        # and Tensor.contiguous has a second input with value 0
        # and torch.arange having a zero input
        def pred(node: Node):
            if node.type == NodeTypes.CONSTANT and (node.value in [0, 1]):
                assert len(node.out_nodes) == 1, "Constant should have one use"
                out = node.out_nodes[0]
                arithmetic_ops = ['aten::add',
                                  'aten::div', 'aten::mul', 'aten::sub']
                arithmetic = any(opMatch(out.scope, o) or opMatch(out.scope, o + "_") for o in arithmetic_ops) and (
                    out.in_nodes.indexOf(node) == 2)
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
        scopeName = extract_new_scope(scopeName)
        for layer_scope in self.profiled_layers:
            if scopeName.startswith(layer_scope):
                most_specific_scope = self.profiled_layers[layer_scope]
                break
        return most_specific_scope

    def _find_partial_match(self, scopeName: str):
        most_specific_scope = ""
        scopeName = extract_new_scope(scopeName)
        for layer_scope in self.partial_scopes:
            if scopeName.startswith(layer_scope) and len(layer_scope) > len(most_specific_scope):
                most_specific_scope = self.partial_scopes[layer_scope]
        return most_specific_scope

    def _remove_nodes_that_go_nowhere(self, trace_graph):
        '''remove nodes without out edges that are not outputs of the model'''
        # necessary because the trace can contain such nodes for certain ops
        # those nodes provide no additional info to the graph
        out_indices = [self._get_id(out) for out in trace_graph.outputs()]

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

    def _get_id(self, out):
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

    def _remove_nodes(self, condition):
        while True:
            changed = False
            optimized_graph = OrderedDict()

            for unique_id, node in self.nodes.items():
                if condition(node):
                    changed = True
                    # connect inputs to outputs directly
                    # TODO we do not remove/add inputs or outputs might revisit
                    for in_node in node.in_nodes:
                        in_node.replace_out_node(node, node.out_nodes)
                        if node.value_type:
                            in_node.value_type = node.value_type
                            in_node.value = None
                    for out_node in node.out_nodes:
                        out_node.replace_in_node(node, node.in_nodes)
                        out_node.inputs.difference_update(node.outputs)
                        out_node.inputs.update(node.inputs)
                else:
                    optimized_graph[unique_id] = node

            self.nodes = optimized_graph
            if not changed:
                break

    def _set_outputs(self, trace_outputs, use_jit_trace):
        outputs = OrderedSet()
        for out in trace_outputs:
            node = out.node()
            scope = self._find_encasing_layer(node.scopeName())
            if scope == '':
                idx = self._get_id(out)
                scope = node.scopeName() + \
                    "/" + node.kind() + str(idx)
                if use_jit_trace and scope.startswith("/"):
                    # TODO this is a bug in jit.trace it works fine with jit.get_trace
                    # weird case where we do not start with model class as prefix
                    scope = self.model_name + scope
            outputs.add(scope)
        self.output_scopes = outputs

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.nodes[key]
        # assume key is scopeName
        for node in self.nodes:
            if node.scope == key:
                return node
        return None

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        discription = ''
        for node in self.nodes:
            discription = f"{discription}\n{node}"
        return discription

    def asNetworkx(self):
        try:
            import networkx as nx
        except ImportError as _:
            print("networkx package not found")
            return

        # edge_list
        edge_list = []
        for u in self.nodes.values():
            for v in u.in_nodes:
                edge_list.append((u.idx, v.idx))

        G = nx.from_edgelist(edge_list)
        for n in self.nodes.values():
            G.nodes[n.idx]['weight'] = n.weight
            G.nodes[n.idx]['scope'] = n.scope
            G.nodes[n.idx]['part'] = n.part

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
                 "padding": "1.0,0.5"}
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

        colors = {0: 'grey', 1: 'green', 2: 'red', 3: 'yellow',
                  4: 'orange', 5: 'brown', 6: 'purple', 7: 'pink'}

        def hide_node(node):
            return (node.type == NodeTypes.BUFF_PARAM) and (not show_buffs_params)

        for node in self.nodes.values():
            if hide_node(node):
                continue
            label = node.scope

            if not node.out_nodes:
                outputs = list(map(str, node.outputs))
                outputs = ",".join(outputs)
                label = f"{label}\n {outputs}"

            if show_weights and node.weight != 0:
                label = f"{label}\n {node.weight}"

            label = f"{label}\n type: {node.valueType()}"
            if not (node.value is None):
                label = f"{label}\n value={node.value}"

            dot.node(str(node.idx), label, fillcolor=colors[node.part])

        for node in self.nodes.values():
            if hide_node(node):
                continue
            for in_node in node.in_nodes:
                if hide_node(in_node):
                    continue
                edge_label = filter(lambda layer_in: layer_in.scope ==
                                    in_node.scope, node.inputs)

                edge_label = list(map(str, edge_label))
                edge_label = ",".join(edge_label)
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
            display_svg(self.build_dot(show_buffs_params,
                                       show_weights=show_weights), raw=False)
        except ImportError as _:
            print("only works in python notebooks")

    def save(self, file_name, directory, show_buffs_params=True, show_weights=False):
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

    @property
    def model_name(self):
        for scope in self.profiled_layers.values():
            return scope.split("/")[0]

    @property
    def num_partitions(self,):
        return len({node.part for node in self.nodes.values()})

    def predecessors(self):
        predecessors = {idx: set() for idx in self.nodes.keys()}

        for idx, n in enumerate(self.nodes.values()):
            for i in n.in_nodes:
                predecessors[idx].update(predecessors[i.idx])
                predecessors[idx].add(i.idx)

        return predecessors


class NodeTypes(Enum):
    '''
    Enum representing the possible types of Nodes in the Graph
    '''
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4
    CONSTANT = 5
    PYTHON_PRIMITIVE = 6

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

    def __init__(self, scope: str, idx: int, node_type: NodeTypes, incoming_nodes=None, weight=0, part=0, value=None):
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
        self.value = value
        self.value_type = None

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

    def replace_out_node(self, to_replace, value):
        if to_replace not in self.out_nodes:
            return

        values = list(self.out_nodes)
        idx = values.index(to_replace)

        before, after = values[:idx], values[idx + 1:]
        try:
            # we handle the case for iterable, if value is not then we recall with [value]
            iter(value)
            keys = value
            to_add = [v for v in keys if (
                v not in before) and (v not in after)]
            self.out_nodes = OrderedSet(before + to_add + after)

        except TypeError as _:
            self.replace_out_node(to_replace, [value])

    def replace_in_node(self, to_replace, value):
        if to_replace not in self.in_nodes:
            return

        values = list(self.in_nodes)
        idx = values.index(to_replace)

        before, after = values[:idx], values[idx + 1:]
        try:
            # we handle the case for iterable, if value is not then we recall with [value]
            iter(value)
            keys = value
            to_add = [v for v in keys if (
                v not in before) and (v not in after)]
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
        if isinstance(other, tuple):
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


def optimize_graph(graph: OLDGraph):
    '''
    this module takes the raw Graph and removes/merges nodes in order to get the requested graph.
    this method is called as part of graph_builder method
    '''
    nodes = graph.nodes
    nodes = _combine_OP_nodes_under_the_same_scope(nodes)
    graph.nodes = nodes
    _combine_params_and_buffers_into_OP_nodes(graph)


def _combine_OP_nodes_under_the_same_scope(nodes: OrderedDict) -> OrderedDict:
    # optimization that reduces number of nodes in the graph
    # combine nodes that have a commom scope we do this because\n
    # if nodes have the same scopeName than they were profiled together
    scope_representative = dict()

    optimized_graph = OrderedDict()

    # get the nodes of the optimized graph
    for unique_id, node in nodes.items():
        if not node.scope in scope_representative:
            optimized_graph[unique_id] = node
            scope_representative[node.scope] = node
        else:
            # add edges create the super set of all edeges in the scope
            scope_representative[node.scope].add_in_node(node.in_nodes)
            scope_representative[node.scope].inputs.update(node.inputs)

            scope_representative[node.scope].add_out_node(node.out_nodes)
            scope_representative[node.scope].outputs.update(node.outputs)

    for node in optimized_graph.values():
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


def _combine_params_and_buffers_into_OP_nodes(graph: OLDGraph):
    def is_buffer_or_param(n):
        return n.type == NodeTypes.BUFF_PARAM and graph._find_encasing_layer(n.scope) != ''

    graph._remove_nodes(is_buffer_or_param)


def opMatch(scope, op_name):
    return re.search(f"{op_name}[{string.digits}]", scope)


def translate_scopes(old_scopes):
    translation = dict()

    pattern = r'\[.*?\]'
    matcher = re.compile(pattern)
    for scope in old_scopes:
        search_results = matcher.finditer(scope)
        translated = ("__module." + ".".join(s.group()
                                             [1:-1] for s in search_results))
        translation[translated] = scope

    return translation


def extract_new_scope(new_scope):
    return new_scope[new_scope.rfind("/__module") + 1:]


def partial_scopes(old_scopes):
    partials = dict()
    for scope in old_scopes:
        base = "__module"
        new_base = ""
        for part in scope.split("/"):
            identifier = part[part.find("[") + 1:-1]
            new_base += f"/{part}"
            partials[base] = new_base[1:]
            base += f".{identifier}"
    return partials


class Graph():
    def __init__(self, nodes, graph_output_scopes, depth, basic_blocks):
        self.nodes = nodes
        self.output_scopes = graph_output_scopes
        self.depth = depth
        self.basic_blocks = basic_blocks

    @property
    def num_inputs(self):
        return len([1 for node in self.nodes.values() if node.type is NodeTypes.IN])

    @property
    def model_name(self):
        for node in self.nodes.values():
            if node.type != NodeTypes.IN:
                return node.scope[:node.scope.find("/")]

    @property
    def num_partitions(self):
        return len(set(node.part for node in self.nodes.values()))

    def asNetworkx(self):
        try:
            import networkx as nx
        except ImportError as _:
            print("networkx package not found")
            return

        # edge_list
        edge_list = []
        for u in self.nodes.values():
            for v in u.in_nodes:
                edge_list.append((u.idx, v.idx))

        G = nx.from_edgelist(edge_list)
        for n in self.nodes.values():
            G.nodes[n.idx]['weight'] = n.weight
            G.nodes[n.idx]['scope'] = n.scope
            G.nodes[n.idx]['part'] = n.part

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
                 "padding": "1.0,0.5"}
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

        colors = {0: 'grey', 1: 'green', 2: 'red', 3: 'yellow',
                  4: 'orange', 5: 'brown', 6: 'purple', 7: 'pink'}

        def hide_node(node):
            return (node.type == NodeTypes.BUFF_PARAM) and (not show_buffs_params)

        for node in self.nodes.values():
            if hide_node(node):
                continue
            label = node.scope

            if not node.out_nodes:
                outputs = list(map(str, node.outputs))
                outputs = ",".join(outputs)
                label = f"{label}\n {outputs}"

            if show_weights and node.weight != 0:
                label = f"{label}\n {node.weight}"

            label = f"{label}\n type: {node.valueType()}"
            if not (node.value is None):
                label = f"{label}\n value={node.value}"

            dot.node(str(node.idx), label, fillcolor=colors[node.part])

        for node in self.nodes.values():
            if hide_node(node):
                continue
            for in_node in node.in_nodes:
                if hide_node(in_node):
                    continue
                edge_label = filter(lambda layer_in: layer_in.scope ==
                                    in_node.scope, node.inputs)

                edge_label = list(map(str, edge_label))
                edge_label = ",".join(edge_label)
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
            display_svg(self.build_dot(show_buffs_params,
                                       show_weights=show_weights), raw=False)
        except ImportError as _:
            print("only works in python notebooks")

    def save(self, file_name, directory, show_buffs_params=True, show_weights=False):
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
