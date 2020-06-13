from enum import IntEnum
import pickle
from typing import Tuple, Optional, Callable, Dict, Iterable, List,Set
from itertools import chain
from torch import Tensor, nn as nn


class NodeTypes(IntEnum):
    '''
    Enum representing the possible types of Nodes in the Graph
    '''
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4
    CONSTANT = 5
    PRIMITIVE = 6

    def __repr__(self):
        return self.name


class Node():
    def __init__(self, node_type, idx, scope):
        self.type = node_type
        self.id = idx
        self.scope = scope

        self.part = -1
        self.weight = None

        self.out_edges:Set[Node] = set()
        self.args = []
        self.kwargs = dict()
        self.value_type = None

        self.tensor_dtype = None
        self.tensor_shape = None
        self.req_grad = False

        self.constant_value = None


    def add_kwarg(self, kwarg, kwarg_node):
        self.kwargs[kwarg_node] = kwarg

    def add_arg(self, arg_node):
        self.args.append(arg_node)

    def add_out_edge(self, dest_node):
        self.out_edges.add(dest_node)

    def remove_output(self, out_node):
        self.out_edges.remove(out_node)

    @property
    def in_edges(self)->List["Node"]:
        return list(chain(self.args, self.kwargs.keys()))

    def replace_input(self,original,new):
        try:
            self.args[self.args.index(original)]=new
        except:
            pass
        if original in self.kwargs:
            self.kwargs[new]=self.kwargs.pop(original)
    
    @classmethod
    def from_other(cls,other):
        node = cls(other.type,other.id,other.scope)
        node.part = other.part
        node.weight = other.weight

        node.out_edges = {n for n in other.out_edges}
        node.args = [n for n in other.in_edges]
        node.kwargs = {n:k for n,k in other.kwargs.items()}
        node.value_type = other.value_type

        node.tensor_dtype = other.tensor_dtype
        node.tensor_shape = other.tensor_shape

        node.constant_value = other.constant_value
        
        return node

GraphNodes = Dict[int, Node]
NodeWeightFunction = Callable[[Node], int]
EdgeWeightFunction = Callable[[Tuple[Node, Node]], int]


class Graph():
    def __init__(self, nodes: GraphNodes, input_kw_ids, output_ids: List[int], depth: int, basic_blocks: Tuple[nn.Module, ...]):
        self._nodes: GraphNodes = nodes
        self.input_kw_ids = input_kw_ids
        self.output_ids = output_ids
        self.depth = depth
        self.basic_blocks = basic_blocks

    def __len__(self)->int:
        return len(self._nodes)
        
    @property
    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    @property
    def inputs(self):
        return (n for n in self.nodes if n.type is NodeTypes.IN)

    @property
    def num_inputs(self) -> int:
        return len(list(self.inputs))

    @property
    def num_partitions(self) -> int:
        return len({n.part for n in self.nodes})

    @property
    def output_scopes(self):
        return [n.scope for n in self.outputs]

    @property
    def outputs(self):
        return [self._nodes[id] for id in self.output_ids]

    @property
    def model_name(self):
        return self._nodes[self.output_ids[0]].scope.split("/", maxsplit=1)[0]

    def asNetworkx(self,
                   directed: bool = False,
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
            whether to return a directed graph default is undirected
        node_weight_function:
            an optional weight function for the nodes should be a function from Node to int
            if not given a default weight of 1 will be given to all nodes
        edge_weight_function:
            an optional weight function for the edges should be a function (Node,Node) to int
            if not given a default value of 1 will be given to all edges
        '''
        try:
            import networkx as nx
        except ImportError:
            print("networkx package not found")
            return

        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for u in self.nodes:
            for v in u.out_edges:
                if edge_weight_function is None:
                    w = 1
                else:
                    w = edge_weight_function(u, v)
                G.add_edge(u.id, v.id, weight=w)

        for n in self.nodes:
            if node_weight_function is None:
                w = 1
            else:
                w = node_weight_function(n)
            G.nodes[n.id]['weight'] = w

        return G

    def build_dot(self,
                  show_buffs_params: bool = True,
                  show_profiles: bool = True,
                  edge_weight_function=None,
                  node_weight_function=None):
        '''
        return a graphviz representation of the graph
        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_profiles:
            whether to display the nodes weight
        edge_weight_function:
            function to get edge weights
        '''
        theme = {
            "background_color": "#FFFFFF",
            "fill_color": "#E8E8E8",
            "outline_color": "#000000",
            "font_color": "#000000",
            "font_name": "Times",
            "font_size": "10",
            "margin": "0,0",
            "padding": "1.0,0.5"
        }
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

        dot.attr("node",
                 shape="box",
                 style="filled",
                 margin="0,0",
                 fillcolor=theme["fill_color"],
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"])

        dot.attr("edge",
                 style="solid",
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"])

        # TODO split big graphs to multiple pdfs

        colors = {
            -1:'grey',
            0: 'grey',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'orange',
            5: 'brown',
            6: 'purple',
            7: 'pink',
            8: 'cyan',
            9: 'gold',
            10: 'darkolivegreen',
            11: 'seagreen',
            12: 'thistle',
            13: 'plum',
            14: 'deeppink',
            15: 'lightyellow',
            16: 'tan'
        }

        def predicate(n):
            return (n.type != NodeTypes.BUFF_PARAM) or show_buffs_params

        # add nodes
        for node in self.nodes:
            node_id = node.id
            if predicate(node):
                scope = node.scope
                value_type = node.value_type
                node_label = f"{scope}\nidx: {node_id}\nvalue type: {value_type}"
                if node in self.outputs:
                    node_label += "\nmodel output"
                if node.type is NodeTypes.IN:
                    node_label += "\nmodel input"
                if node.id in self.input_kw_ids:
                    node_label += f"\nkwarg: {self.input_kw_ids[node.id]}"
                if node.type is NodeTypes.CONSTANT:
                    node_label += f"\nvalue: {node.constant_value}"

                if issubclass(node.value_type, Tensor):
                    node_label += f"\ntensor of type: {node.tensor_dtype}\nshape: {node.tensor_shape}"

                if show_profiles and node.weight:
                    node_label = f"{node_label}\nProfile:"
                    for k, v in node.weight._asdict().items():
                        node_label += f"\n{k}:{v}"
                        if "time" in k:
                            node_label += " ms"
                        elif "memory" in k or "size" in k:
                            node_label += " MB"
                if node_weight_function:
                    node_label += f"\nweight: {node_weight_function(node)}"

                dot.node(str(node_id), label=node_label,
                         fillcolor=colors[node.part])

                # add edges
                args, kwargs = node.args, node.kwargs
                for idx, i in enumerate(args):
                    if predicate(i):
                        edge_label = f"arg: {idx}"
                        if edge_weight_function:
                            edge_label += f"\nweight: {edge_weight_function(i,node)}"
                        dot.edge(str(i.id), str(node_id), label=edge_label)

                for i, kw in kwargs.items():
                    if predicate(i):
                        edge_label = f"kwarg: {kw}"
                        if edge_weight_function:
                            edge_label += f"\nweight: {edge_weight_function(i,node)}"
                        dot.edge(str(i.id), str(node_id), label=edge_label)

        return dot

    def display(self,
                show_buffs_params: bool = True,
                show_profiles: bool = True,
                edge_weight_function=None):
        '''
        display the graph in Jupyter

        Parameters
        ----------
        show_buffs_params:
            whether to display also buffers and parameters which are not encased in the graph scopes
        show_profiles:
            whether to display the nodes weight
        edge_weight_function:
            edge weight function to use
        '''
        try:
            from IPython.core.display import display_svg
            display_svg(self.build_dot(show_buffs_params,
                                       show_profiles=show_profiles,
                                       edge_weight_function=edge_weight_function),
                        raw=False)
        except ImportError:
            print("only works in ipython notebooks")

    def save_as_pdf(self,
                    file_name: str,
                    directory: str,
                    show_buffs_params: bool = True,
                    show_profiles: bool = True,
                    edge_weight_function=None,
                    node_weight_function=None):
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
        dot = self.build_dot(show_buffs_params, show_profiles=show_profiles,
                             edge_weight_function=edge_weight_function, node_weight_function=node_weight_function)
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
        if not path.endswith(".graph"):
            path += ".graph"

        pickle.dump(self.state(), open(path, "wb"))

    def state(self):
        '''
        returns a dicitionary containing the graphs state
        '''

        node_states = []
        for node in self.nodes:
            state = dict(id=node.id,
                        scope=node.scope,
                        type=node.type,
                        part=node.part,
                        weight=node.weight,
                        out_edges=[n.id for n in node.out_edges],
                        args=[n.id for n in node.args],
                        kwargs={n.id: kw for n,
                                kw in node.kwargs.items()},
                        value_type=node.value_type,
                        constant_value=node.constant_value,
                        tensor_dtype=node.tensor_dtype,
                        tensor_shape=node.tensor_shape,
                        req_grad=node.req_grad)
            node_states.append(state)

        return{"node_data": node_states,
               "input_kw_ids": self.input_kw_ids,
               "output_ids": self.output_ids,
               "depth": self.depth,
               "basic_blocks": self.basic_blocks
               }

    def load_state(self, state):
        output_ids = state['output_ids']
        depth = state['depth']
        basic_blocks = state['basic_blocks']
        input_kw_ids = state['input_kw_ids']

        nodes = dict()

        states = state['node_data']
        for state in states:
            node = Node(state['type'], state['id'], state['scope'])
            nodes[node.id] = node

            node.part = state['part']
            node.weight = state['weight']
            node.args = [nodes[n] for n in state['args']]
            node.kwargs = {nodes[n]: kw for n, kw in state['kwargs'].items()}
            node.constant_value = state['constant_value']
            node.value_type = state['value_type']
            node.tensor_dtype = state['tensor_dtype']
            node.tensor_shape = state['tensor_shape']
            node.req_grad = state['req_grad']

        for node in nodes.values():
            node.out_edges = {nodes[n] for n in states[node.id]['out_edges']}

        self._nodes = nodes
        self.basic_blocks = basic_blocks
        self.depth = depth
        self.output_ids = output_ids
        self.input_kw_ids = input_kw_ids

        return self

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

        return cls(None, None, None, None, None).load_state(graph_data)

    def layers_graph(self) -> Tuple["Graph", Dict[int, int]]:
        '''
        creates a graph g with nodes of type CONSTANT 
        or nodes who soley depend on constants are removed
        leaving only inputs layers and params/buffers

        returns the created graph and a map between g's indices and self indices
        '''
        new_nodes = dict()
        output_ids = []

        new_graph = Graph(None, None, None, None,
                          None).load_state(self.state())

        num_removed = 0
        lookup = dict()
        for node in new_graph._nodes.values():
            if node.type is NodeTypes.CONSTANT or ((node.type in [NodeTypes.PRIMITIVE,NodeTypes.OP]) and (len(node.in_edges) == 0)):
                for o in node.out_edges:
                    o.kwargs.pop(node, None)
                    o.args = [n for n in o.args if n is not node]
                num_removed += 1
            else:
                old_id = node.id
                new_id = old_id - num_removed
                if node.id in new_graph.output_ids:
                    output_ids.append(new_id)
                node.id = new_id
                new_nodes[new_id] = node
                lookup[new_id] = old_id

        new_graph._nodes = new_nodes
        new_graph.output_ids = output_ids

        return new_graph, lookup

    def __getitem__(self,idx):
        return self._nodes[idx]