from enum import Enum
from typing import Any, List, Tuple, Optional, OrderedDict, Type, Callable, Union
from ..utils import OrderedSet
from .network_profiler import Profile
import torch
from torch.nn import Module
from torch import Tensor
import pickle
import collections


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
    value:
        if the node is of type NodeType.CONSTANT then value is the constant this node represents
    incoming_nodes:
        the nodes who have edges from them to this node
    out_nodes:
        the nodes who have edges from this node
    weight:
        the weight of the edge can be anything
    part:
        partition idx determines the color of the Node

     parallel edges in the same direction are not allowed
    '''

    def __init__(self, scope: str, idx: int, node_type: NodeTypes, incoming_nodes: Optional[OrderedSet["Node"]] = None, weight: Union[Profile, int] = 0, part: int = 0, value: Optional[Any] = None, shape: Optional[List[int]] = None):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes: OrderedSet[Node] = OrderedSet()
        self.weight = weight
        self.part = part
        self.in_nodes: OrderedSet[Node] = incoming_nodes if isinstance(
            incoming_nodes, OrderedSet) else OrderedSet()
        self.value = value
        self.value_type: Optional[Type] = None
        self.shape = [] if shape is None else shape

    def valueType(self) -> Type:
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


GraphNodes = OrderedDict[int, Node]
NodeWeightFunction = Callable[[Node], int]
EdgeWeightFunction = Callable[[Tuple[Node, Node]], int]


class Graph():
    '''
    A simple graph data structure to be consumed by the partition algorithm and the code generator

    Fields:
    ------
    nodes: List[Node]:
        the graph Nodes
    output_scopes: OrderedSet[str]:
        the graph's output scopes representing the model outputs
    depth: int:
        the depth used in order to construct the graph
    basic_blocks: Tuple[Module,...]:
        the basic blocks specified by the user at creation time
    num_inputs: int:
        the number of inputs(Tensors) the model recieves
    model_name: str:
        the name of the model class this graph represents
    num_partitions: int:
        the number of partitions for the nodes
    '''

    def __init__(self, nodes: GraphNodes, graph_output_scopes: OrderedSet[str], depth: int, basic_blocks: Tuple[Module, ...]):
        self._nodes = nodes
        self.output_scopes = graph_output_scopes
        self.depth = depth
        self.basic_blocks = basic_blocks

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def num_inputs(self) -> int:
        return len([1 for node in self.nodes if node.type is NodeTypes.IN])

    @property
    def model_name(self) -> str:
        for node in self.nodes:
            if node.type != NodeTypes.IN:
                return node.scope[:node.scope.find("/")]

    @property
    def num_partitions(self) -> int:
        return len(set(node.part for node in self.nodes))

    def asNetworkx(self, directed: bool = False,
                   node_weight_function: Optional[NodeWeightFunction] = None,
                   edge_weight_function: Optional[EdgeWeightFunction] = None):
        '''
        convert the graph into a weighted networkx graph.\n
        each node will have a scope,partition idx and weight associated with it.\n
        each weight will be weighted\n
        graph can be directed or undirected for a directed graph weighting functions can be given
        if not then weight will be set to 1.\n

        Parameters:
        ------------
        directed:
            wether to return a directed graph default is undirected
        node_weight_function:
            an optional weight function for the nodes should be a function from Node to int
            if not given a default weight of 1 will be given to all nodes
        edge_weight_function:
            an optional weight function for the edges should be a function (Node,Node) to int
            if not given a default value of 1 will be given to all edges
        '''
        try:
            import networkx as nx
        except ImportError as _:
            print("networkx package not found")
            return

        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for u in self.nodes:
            for v in u.out_nodes:
                if edge_weight_function is None:
                    w = 1
                else:
                    w = edge_weight_function(u, v)
                G.add_edge(u.idx, v.idx, weight=w)

        for n in self.nodes:
            if node_weight_function is None:
                w = 1
            else:
                w = node_weight_function(n)
            G.nodes[n.idx]['weight'] = w
            G.nodes[n.idx]['label'] = n.scope
            G.nodes[n.idx]['partition_idx'] = n.part
            G.nodes[n.idx]['original_profile'] = n.weight

        return G

    def build_dot(self, show_buffs_params: bool = False, show_profiles: bool = True):
        '''
        return a graphviz representation of the graph
        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_profiles:
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

        for node in self.nodes:
            if hide_node(node):
                continue
            label = node.scope

            if show_profiles and node.weight != 0:
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
                dot.edge(str(in_node.idx), str(node.idx))

        return dot

    def display(self, show_buffs_params: bool = False, show_profiles: bool = True):
        '''
        display the graph in Jupyter

        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_profiles:
            whether to display the nodes weight
        '''
        try:
            from IPython.core.display import display_svg
            display_svg(self.build_dot(show_buffs_params,
                                       show_profiles=show_profiles), raw=False)
        except ImportError as _:
            print("only works in python notebooks")

    def save_as_pdf(self, file_name: str, directory: str, show_buffs_params: bool = True, show_profiles: bool = False):
        '''
        save the rendered graph to a pdf file

        Parameters
        ----------
        file_name:
            the name of the saved file
        directory:
            directory to store the file in
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_profiles:
            whether to display the nodes weight
        '''
        dot = self.build_dot(show_buffs_params, show_profiles=show_profiles)
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)
        return self

    def serialize(self, path: str):
        '''
        serializes the graph to the given path
        can later be restored using Graph.deserialize(path)

        Parameters:
        -----------
        path:
            the path to store the graph object file will be called path.graph
        '''
        path += ".graph"

        graph_output_scopes = self.output_scopes
        graph_depth = self.depth
        graph_basic_blocks = self.basic_blocks
        graph_nodes_data = []
        for u in self.nodes:
            in_nodes = [v.idx for v in u.in_nodes]
            out_nodes = [v.idx for v in u.out_nodes]
            node_data = {"idx": u.idx, "part": u.part, "weight": u.weight,
                         "scope": u.scope, "type": u.type, "value": u.value,
                         "value_type": u.value_type,
                         "in_nodes": in_nodes, "out_nodes": out_nodes}
            graph_nodes_data.append(node_data)

        graph = {"depth": graph_depth,
                 "output_scopes": graph_output_scopes,
                 "basic_blocks": graph_basic_blocks,
                 "nodes_data": graph_nodes_data}

        pickle.dump(graph, open(path, "wb"))

    def graphs_equal(self, other) -> bool:
        '''
        check if 2 graphs are equal\n
        graphs are equal if the topography is the same and if the nodes data is the same (upto weights and partition idx)
        '''
        if not isinstance(other, Graph):
            return False

        if len(self.nodes) != len(other.nodes):
            return False

        for u, v in zip(self.nodes, other.nodes):
            if u.idx != v.idx or u.scope != v.scope or u.type != v.type:
                return False
            if u.value_type != v.value_type or u.valueType() != v.valueType():
                return False

            if u.valueType() is Tensor:
                if (u.value is None or v.value is None) and (not (u.value is v.value)):
                    return False
                if (u.value != None and v.value != None) and (not torch.allclose(u.value, v.value)):
                    return False
            if u.valueType() != Tensor and u.value != v.value:
                return False

            if len(u.out_nodes) != len(v.out_nodes):
                return False
            for x, y in zip(u.out_nodes, v.out_nodes):
                if x.idx != y.idx:
                    return False

            if len(u.in_nodes) != len(v.in_nodes):
                return False
            for x, y in zip(u.in_nodes, v.in_nodes):
                if x.idx != y.idx:
                    return False

        return True

    @classmethod
    def deserialize(cls, path: str) -> "Graph":
        '''
        deserializes the graph from the path returning a Graph object

        Parameters:
        -------------
        path:
        the path to where the graph is stored
        '''
        if not path.endswith(".graph"):
            path += ".graph"

        graph_data = pickle.load(open(path, "rb"))
        nodes = collections.OrderedDict()

        # load node data
        for node in graph_data["nodes_data"]:
            idx = node["idx"]
            part = node["part"]
            weight = node["weight"]
            scope = node["scope"]
            node_type = node["type"]
            value = node["value"]
            value_type = node["value_type"]
            nodes[idx] = Node(scope, idx, node_type,
                              weight=weight, part=part, value=value)
            nodes[idx].value_type = value_type

        # add edges
        for node in graph_data["nodes_data"]:
            nodes[node["idx"]].in_nodes = OrderedSet([nodes[u]
                                                      for u in node["in_nodes"]])
            nodes[node["idx"]].out_nodes = OrderedSet([nodes[u]
                                                       for u in node["out_nodes"]])

        return cls(nodes, graph_data["output_scopes"], graph_data["depth"], graph_data["basic_blocks"])

    @classmethod
    def _check(cls, nodes_or_graph: Union[GraphNodes, "Graph"]):
        if isinstance(nodes_or_graph, Graph):
            nodes = nodes_or_graph.nodes
            _nodes = nodes_or_graph._nodes
        else:
            assert isinstance(nodes_or_graph, OrderedDict)
            nodes = list(nodes_or_graph.values())
            _nodes = nodes_or_graph

        assert isinstance(_nodes, OrderedDict)
        assert isinstance(nodes[0], Node)
        for node in nodes:
            assert node.idx in _nodes
            for i in node.in_nodes:
                assert node in i.out_nodes
                assert i.idx in _nodes
            for o in node.out_nodes:
                assert node in o.in_nodes
                assert o.idx in _nodes
        return nodes_or_graph
