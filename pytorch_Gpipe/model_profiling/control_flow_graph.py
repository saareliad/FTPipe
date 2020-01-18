from enum import Enum
from typing import Any, Union, List, Tuple, Optional, OrderedDict, Type
from ..utils import OrderedSet
from .network_profiler import Profile
from torch.nn import Module
# TODO support list and tuple layer outputs


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

    def __init__(self, scope: str, idx: int, node_type: NodeTypes, incoming_nodes: Optional[OrderedSet["Node"]] = None, weight: Union[Profile, int] = 0, part: int = 0, value: Optional[Any] = None):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes = OrderedSet()
        self.weight = weight
        self.part = part
        self.in_nodes = incoming_nodes if isinstance(
            incoming_nodes, OrderedSet) else OrderedSet()
        self.value = value
        self.value_type: Optional[Type] = None

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

    def asNetworkx(self):
        try:
            import networkx as nx
        except ImportError as _:
            print("networkx package not found")
            return

        # edge_list
        edge_list = []
        for u in self.nodes:
            for v in u.in_nodes:
                edge_list.append((u.idx, v.idx))

        G = nx.from_edgelist(edge_list)
        for n in self.nodes:
            G.nodes[n.idx]['weight'] = n.weight
            G.nodes[n.idx]['scope'] = n.scope
            G.nodes[n.idx]['part'] = n.part

        return G

    def build_dot(self, show_buffs_params: bool = False, show_weights: bool = True):
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

        for node in self.nodes:
            if hide_node(node):
                continue
            label = node.scope

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
                dot.edge(str(in_node.idx), str(node.idx))

        return dot

    def display(self, show_buffs_params: bool = False, show_weights: bool = True):
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

    def save(self, file_name: str, directory: str, show_buffs_params: bool = True, show_weights: bool = False):
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
