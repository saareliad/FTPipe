import pickle
import warnings
from collections import defaultdict
from copy import deepcopy
from enum import IntEnum
from itertools import chain
from typing import Tuple, Optional, Callable, Dict, Iterable, List, Type, Set

import networkx as nx
from torch import Tensor, nn as nn

from autopipe.autopipe.union_find import UnionFind
from autopipe.autopipe.utils import ExecTimes, layerDict, tensorDict

from torch.nn import Parameter, Module

class NodeTypes(IntEnum):
    """
    Enum representing the possible types of Nodes in the Graph
    """
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4
    CONSTANT = 5
    PRIMITIVE = 6

    def __repr__(self):
        return self.name


class Node:
    def __init__(self, node_type, idx, scope):
        self.type = node_type
        self.id = idx
        self.scope = scope

        self.topo_sort_id = id  # hack: start with ID. (potential danger)
        self.stage_id = 0
        self.gpu_id = None  # New feature
        self.weight: Optional[ExecTimes] = None
        self.max_memory_bytes: float = 0
        self.num_parameters: Optional[int] = 0
        self.compound_edge_weights = defaultdict(float)
        self.out_edges: List[Node] = []
        self.args = []
        self.kwargs = defaultdict(list)
        self.value_type = None

        self.tensor_dtype = None
        self.tensor_shape = None
        self.req_grad = False

        self.constant_value = None

    # def __repr__(self):
    #     return self.scope

    def maybe_create_compound_edge_weights(self, edge_weight_function):
        if not self.compound_edge_weights:
            for nn in self.out_edges:
                self.compound_edge_weights[nn.id] = edge_weight_function(self, nn)
            return True if self.out_edges else False
        return False

    def update_compound_weights_from_uf(self, uf: UnionFind, allow_outside_uf=False):

        items_to_handle = list(self.compound_edge_weights.items())
        if allow_outside_uf:
            items_to_handle = filter(lambda x: x[0] in uf, items_to_handle)

        for id, weight in items_to_handle:
            if not uf.is_root(id):
                new_id = uf[uf.find(id)]
                if new_id != id:
                    del self.compound_edge_weights[id]
                    self.compound_edge_weights[new_id] = weight

    def add_kwarg(self, kwarg, kwarg_node):
        self.kwargs[kwarg_node].append(kwarg)

    def add_arg(self, arg_node):
        self.args.append(arg_node)

    def add_out_edge(self, dest_node):
        self.out_edges.append(dest_node)

    def remove_output(self, out_node):
        self.out_edges.remove(out_node)

    def replace_out_edge_on_merge(self, dest_node, new_dest_node):
        self.remove_output(dest_node)
        self.out_edges.append(new_dest_node)

    @property
    def in_edges(self) -> List["Node"]:
        return list(chain(self.args, self.kwargs.keys()))

    def replace_input(self, original, new):
        must_be_in_kwargs = False
        try:
            self.args[self.args.index(original)] = new
        except:
            must_be_in_kwargs = True

        if original in self.kwargs:
            self.kwargs[new] = self.kwargs.pop(original)
        elif must_be_in_kwargs:
            raise KeyError(f"original is not in args and not in kwargs, original={original}\nself={self}")

    def remove_input(self, original):
        must_be_in_kwargs = False
        try:
            del self.args[self.args.index(original)]
        except:
            must_be_in_kwargs = True
        if original in self.kwargs:
            self.kwargs.pop(original)
        elif must_be_in_kwargs:
            raise KeyError(f"original is not in args and not in kwargs, original={original}")

    @classmethod
    def from_other(cls, other):
        node = cls(other.type, other.id, other.scope)
        node.stage_id = other.stage_id
        node.gpu_id = other.gpu_id
        node.weight = other.weight
        node.num_parameters = other.num_parameters

        node.topo_sort_id = other.topo_sort_id
        node.out_edges = list(other.out_edges)
        node.args = list(other.args)
        node.kwargs = dict(other.kwargs)
        node.value_type = other.value_type

        node.tensor_dtype = other.tensor_dtype
        node.tensor_shape = other.tensor_shape
        node.req_grad = other.req_grad

        node.constant_value = other.constant_value
        node.max_memory_bytes = other.max_memory_bytes

        return node

    def __repr__(self):
        return str(f"id: {self.id}, scope:{self.scope}, type:{self.type}")


GraphNodes = Dict[int, Node]
NodeWeightFunction = Callable[[Node], int]
EdgeWeightFunction = Callable[[Node, Node], int]


class Graph():
    def __init__(self, nodes: Optional[GraphNodes], input_kw_ids: Optional[Dict[int, str]],
                 output_ids: Optional[List[int]], depth: Optional[int],
                 basic_blocks: Optional[Tuple[Type[nn.Module], ...]]):
        # TODO: created in trace module, take doc from there.
        self._nodes: GraphNodes = nodes
        self.input_kw_ids = input_kw_ids  # id to kw name.
        self.output_ids = output_ids
        self.depth = depth
        self.basic_blocks = basic_blocks

    def merge(self, uid: int, vid: int, edge_weight_function: EdgeWeightFunction, dynamic_topo_sort=False,
              uf: Optional[UnionFind] = None, partial_uf: Optional[UnionFind] = None):
        if vid in self.output_ids:
            self.output_ids[self.output_ids.index(vid)] = uid
        assert uid != vid
        # print(f"merge {(uid,vid)} ")
        # TODO: check if its legal merge. currently its user's responsibility.
        # does not exits a path from the set {n : u < n < v} to v.
        u = self._nodes[uid]
        v = self._nodes[vid]

        u.remove_output(v)
        v.remove_input(u)

        self._update_compound_weights_on_merge(u, v, edge_weight_function, uf, partial_uf)

        for a in v.in_edges:
            a.maybe_create_compound_edge_weights(edge_weight_function=edge_weight_function)


        # Merge node weights
        # TODO: we don't know about other x->v or u->y
        # Needs outside control to with longest path to make it accurate in case of concurrent ops
        # (there is a path u->v hence its sequential).
        weight = ExecTimes(u.weight.forward_time + v.weight.forward_time,
                           u.weight.backward_time + v.weight.backward_time
                           )
        # u.weight.forward_time += v.weight.forward_time
        # u.weight.backward_time += v.weight.backward_time
        u.weight = weight

        # Do the coarsening
        for nn in v.out_edges:
            u.out_edges.append(nn)
            nn.replace_input(v, u)
            nn.args = remove_dups(nn.args, nn)

        for nn in v.args:
            nn.remove_output(v)
            nn.add_out_edge(u)
            u.args.append(nn)

            nn.out_edges = remove_dups(nn.out_edges, nn)

        # shouldn't have kwargs anyway this is redundant.
        for nn, nnv in v.kwargs.values():
            nn.remove_output(v)
            nn.add_out_edge(u)
            nn.out_edges = remove_dups(nn.out_edges, nn)
            u.kwargs[nn] = nnv

        u.args = remove_dups(u.args, u)
        u.out_edges = remove_dups(u.out_edges, u)
        del self._nodes[vid]

        # merge node parameters
        u.num_parameters += v.num_parameters
        u.max_memory_bytes += v.max_memory_bytes

        # TODO: dynamic topo_sort
        if dynamic_topo_sort:
            raise NotImplementedError()

        # self.topo_sort(verbose=False, change_graph=False)

        # # Massive check for dups....
        # for n in self.nodes:
        #     assert n not in n.out_edges, n
        #     assert n not in n.in_edges, n
        #     assert len(set(n.out_edges)) == len(n.out_edges), n
        #     assert len(set(n.in_edges)) == len(n.in_edges), n

    def _update_compound_weights_on_merge(self, u, v, edge_weight_function, uf, partial_uf: Optional[UnionFind] = None):
        # Merge edge weights.
        is_new_u_weights = u.maybe_create_compound_edge_weights(edge_weight_function)
        # TODO: if is_new_u_weights remove v from there, but the id is unkown at this step.
        is_new_v_weights = v.maybe_create_compound_edge_weights(edge_weight_function)
        if uf is None:
            warnings.warn("merge without union find may miss compound edge weights")
        else:
            # (1): update the current compound weight ids according to the unions so far.
            # (2): add
            if not is_new_v_weights:
                v.update_compound_weights_from_uf(uf)
                if partial_uf:
                    v.update_compound_weights_from_uf(partial_uf, allow_outside_uf=True)
            if not is_new_u_weights:
                u.update_compound_weights_from_uf(uf)
                if partial_uf:
                    u.update_compound_weights_from_uf(partial_uf, allow_outside_uf=True)

        for id, weight in v.compound_edge_weights.items():
            u.compound_edge_weights[id] += weight

        for a in v.in_edges:
            is_new_a_weights = a.maybe_create_compound_edge_weights(edge_weight_function)
            if not is_new_a_weights:
                a.update_compound_weights_from_uf(uf)
                if partial_uf:
                    a.update_compound_weights_from_uf(partial_uf, allow_outside_uf=True)

            if u in a.out_edges:
                w = a.compound_edge_weights[u.id]
                del a.compound_edge_weights[u.id]
                a.compound_edge_weights[v.id] += w  # defaultdict

    #
    # def _update_compound_weights_on_replaced_output(self, nn, u, v, edge_weight_function, uf):

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def n_stages(self) -> int:
        return len({n.stage_id for n in self.nodes})

    @property
    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    @property
    def non_input_nodes(self) -> Iterable[Node]:
        for n in self._nodes.values():
            if n.type is NodeTypes.IN:
                continue
            yield n

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def inputs(self):
        return (n for n in self.nodes if n.type is NodeTypes.IN)

    @property
    def num_inputs(self) -> int:
        return len(list(self.inputs))

    @property
    def num_partitions(self) -> int:
        return len({n.stage_id for n in self.nodes})

    @property
    def unique_partitions_ids(self):
        return {n.stage_id for n in self.nodes}

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
                   edge_weight_function: Optional[EdgeWeightFunction] = None) -> nx.Graph:
        """
        convert the graph into a weighted networkx graph.\n
        each node will have a scope,partition idx and weight associated with it.\n
        each weight will be weighted\n
        graph can be directed or undirected for a directed graph weighting functions can be given
        parallel edges will be discarded\n
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
        """
        try:
            import networkx as nx
        except ImportError as e:
            print("networkx package not found")
            raise e

        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for u in self.nodes:
            dsts = set()
            for v in u.out_edges:
                # disallow parallel in edges
                # disallow parllel out edges
                if v.id in dsts:
                    continue
                dsts.add(v.id)
                if edge_weight_function is None:
                    w = 1
                else:
                    w = int(max(1, edge_weight_function(u, v)))
                G.add_edge(u.id, v.id, weight=w)

        for n in self.nodes:
            if node_weight_function is None:
                w = 1
            else:
                w = node_weight_function(n)
            G.nodes[n.id]['weight'] = int(w)

        return G

    def build_dot(self,
                  node_weight_function: Optional[NodeWeightFunction] = None,
                  edge_weight_function: Optional[EdgeWeightFunction] = None):
        """
        return a graphviz representation of the graph
        Parameters
        ----------
        node_weight_function:
            optional function to get node weights
        edge_weight_function:
            optional function to get edge weights
        """
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

        colors = {
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

        # add nodes
        for node in self.nodes:
            node_id = node.id

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

                if node.weight:
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
                     fillcolor=colors[node.stage_id])

            # add edges
            args, kwargs = node.args, node.kwargs
            for idx, i in enumerate(args):
                edge_label = f"arg: {idx}"
                if edge_weight_function:
                    edge_label += f"\nweight: {edge_weight_function(i, node)}"
                dot.edge(str(i.id), str(node_id), label=edge_label)

            for i, kws in kwargs.items():
                for kw in kws:
                    edge_label = f"kwarg: {kw}"
                    if edge_weight_function:
                        edge_label += f"\nweight: {edge_weight_function(i, node)}"
                    dot.edge(str(i.id), str(node_id), label=edge_label)

        return dot

    def display(self,
                node_weight_function: Optional[NodeWeightFunction] = None,
                edge_weight_function: Optional[EdgeWeightFunction] = None):
        """
        display the graph in Jupyter

        Parameters
        ----------
        edge_weight_function:
            optional edge weight function
        node_weight_function:
            optional node weight function
        """
        try:
            from IPython.core.display import display_svg
            display_svg(self.build_dot(node_weight_function=node_weight_function,
                                       edge_weight_function=edge_weight_function),
                        raw=False)
        except ImportError:
            print("only works in ipython notebooks")

    def save_as_pdf(self,
                    file_name: str,
                    directory: str,
                    node_weight_function: Optional[NodeWeightFunction] = None,
                    edge_weight_function: Optional[EdgeWeightFunction] = None):
        """
        save the rendered graph to a pdf file

        Parameters
        ----------
        file_name:
            the name of the saved file
        directory:
            directory to store the file in
        """
        dot = self.build_dot(edge_weight_function=edge_weight_function, node_weight_function=node_weight_function)
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)
        return self

    def serialize(self, path: str):
        """
        serializes the graph to the given path
        can later be restored using Graph.deserialize(path)

        Parameters:
        -----------
        path:
            the path to store the graph object file will be called path.graph
        """

        with open(path, "wb") as f:
            pickle.dump(self.state(), f)

    def state(self):
        """
        returns a dictionary containing the graphs state
        """
        node_states = dict()
        for node in self.nodes:
            state = dict(id=node.id,
                         scope=node.scope,
                         type=node.type,
                         topo_sort_id=node.topo_sort_id,
                         stage_id=node.stage_id,
                         gpu_id=node.gpu_id,
                         weight=node.weight,
                         num_parameters=node.num_parameters,
                         out_edges=[n.id for n in node.out_edges],
                         args=[n.id for n in node.args],
                         kwargs={n.id: kw for n,
                                              kw in node.kwargs.items()},
                         value_type=node.value_type,
                         constant_value=node.constant_value,
                         tensor_dtype=node.tensor_dtype,
                         tensor_shape=node.tensor_shape,
                         req_grad=node.req_grad,
                         compound_edge_weights=deepcopy(node.compound_edge_weights),
                         max_memory_bytes=node.max_memory_bytes)
            node_states[node.id] = state

        return {"node_data": node_states,
                "input_kw_ids": self.input_kw_ids,
                "output_ids": self.output_ids,
                "depth": self.depth,
                "basic_blocks": self.basic_blocks
                }

    def load_state(self, graph_state):
        output_ids = graph_state['output_ids']
        depth = graph_state['depth']
        basic_blocks = graph_state['basic_blocks']
        input_kw_ids = graph_state['input_kw_ids']

        nodes = dict()
        node_states = graph_state['node_data']
        for state in sorted(node_states.values(), key=lambda s: s['id']):
            node = Node(state['type'], state['id'], state['scope'])
            nodes[node.id] = node

        for state in sorted(node_states.values(), key=lambda s: s['id']):
            node = nodes[state['id']]

            node.topo_sort_id = state['topo_sort_id']
            node.stage_id = state['stage_id']
            node.gpu_id = state['gpu_id']
            node.weight = state['weight']
            node.args = [nodes[n] for n in state['args']]  # NOTE: must be same order!
            node.kwargs = {nodes[n]: kw for n, kw in state['kwargs'].items()}
            node.constant_value = state['constant_value']
            node.value_type = state['value_type']
            node.tensor_dtype = state['tensor_dtype']
            node.tensor_shape = state['tensor_shape']
            node.req_grad = state['req_grad']
            node.compound_edge_weights = state['compound_edge_weights']
            node.num_parameters = state['num_parameters']
            node.max_memory_bytes = state['max_memory_bytes']

        for node in nodes.values():
            node.out_edges = sorted({nodes[n] for n in node_states[node.id]['out_edges']}, key=lambda x: x.id)

        self._nodes = nodes
        self.basic_blocks = basic_blocks
        self.depth = depth
        self.output_ids = output_ids
        self.input_kw_ids = input_kw_ids

        return self

    @classmethod
    def deserialize(cls, path: str) -> "Graph":
        """
        deserializes the graph from the path returning a Graph object

        Parameters:
        -------------
        path:
        the path to where the graph is stored
        """

        with open(path, "rb") as f:
            graph_data = pickle.load(f)

        return cls(None, None, None, None, None).load_state(graph_data)

    @classmethod
    def from_state(cls, state) -> "Graph":
        return cls(None, None, None, None, None).load_state(state)

    @classmethod
    def from_other(cls, graph: "Graph") -> "Graph":
        return cls(None, None, None, None, None).load_state(graph.state())

    def layers_graph(self) -> Tuple["Graph", Dict[int, int]]:
        """
        creates a graph g with nodes of type CONSTANT 
        or nodes who solely depend on constants are removed
        leaving only inputs layers and params/buffers

        returns the created graph and a map between g's indices and self indices
        """
        new_nodes = dict()
        output_ids = []

        new_graph = Graph.from_other(self)

        num_removed = 0
        lookup = dict()
        for node in new_graph._nodes.values():
            is_constant = node.type is NodeTypes.CONSTANT
            op_without_inputs = (node.type in [NodeTypes.PRIMITIVE, NodeTypes.OP]) and (len(node.in_edges) == 0)

            # NOTE i hate this but it handles passing labels which are used only at last partition
            # FIXME this /2 is problematic since we topo sort.
            input_or_buff_param_with_one_use_at_end = (node.type in [NodeTypes.IN, NodeTypes.BUFF_PARAM]) and (
                    len(node.out_edges) == 1)
            if input_or_buff_param_with_one_use_at_end:
                input_or_buff_param_with_one_use_at_end &= (list(node.out_edges)[0].id - node.id) >= (len(self) / 2)

            if is_constant or op_without_inputs or input_or_buff_param_with_one_use_at_end:
                if input_or_buff_param_with_one_use_at_end and not (is_constant or op_without_inputs):
                    warnings.warn(f"HACK: input_or_buff_param_with_one_use_at_end=True, for {node.scope},"
                                  f"list(node.out_edges)[0].id={list(node.out_edges)[0].id} "
                                  f"node.id={node.id}, "
                                  f"len(self)={len(self)}")
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

    def induce_layer_partition(self, layers_graph,
                               layers_to_original: Dict[int, int]) -> "Graph":
        assert len(self) >= len(layers_graph)
        old_to_new = {v: k for k, v in layers_to_original.items()}
        # iterate in reverse order
        for node in sorted(self.nodes, key=lambda n: n.id, reverse=True):
            if node.id in old_to_new:
                take_from = layers_graph[old_to_new[node.id]]
                node.stage_id = take_from.stage_id
                node.gpu_id = take_from.gpu_id
            else:
                # as we iterate in reverse topological order we've already handled this node's outputs
                # select the lowest partition index to ensure no cycles are created
                first = sorted(node.out_edges, key=lambda n: n.stage_id)[0]
                node.stage_id = first.stage_id
                node.gpu_id = first.gpu_id
            assert node.stage_id >= 0

        return self

    def __getitem__(self, idx):
        return self._nodes[idx]

    def __contains__(self, node_id):
        return node_id in self._nodes

    def selfcheck(self):
        visited = set()
        try:
            for n in self.nodes:
                for u in n.in_edges:
                    assert u.id < n.id
                    assert n in u.out_edges, (n.scope, u.scope)
                    visited.add(u)
                assert n not in n.in_edges
                for o in n.out_edges:
                    assert o.id > n.id
                    assert n in o.in_edges, (n.scope, o.scope)
                    visited.add(o)
                visited.add(n)
                assert n not in n.out_edges
            assert len(visited) == self.num_nodes
        except AssertionError as e:
            self.save_as_pdf("selfcheck_error", ".")
            raise e
        return self

    def split_to_stages(self) -> Dict[int, "Graph"]:
        """return a sub graph for each stage in the graph

        Returns:
            Dict[int,Graph] 
        """
        stages = dict()

        tmp = Graph(None, None, None, None, None).load_state(self.state())

        groups = defaultdict(list)
        for n in tmp.nodes:
            if n.type != NodeTypes.IN:
                groups[n.stage_id].append(n)

        for stage_id, group in groups.items():
            stage_nodes = dict()
            stage_inputs = dict()
            stage_output_ids = []
            stage_input_kws = dict()

            for n in sorted(group, key=lambda w: w.id):
                stage_nodes[n.id] = n
                # check if stage output
                if (n.id in self.output_ids) or any(o.stage_id != stage_id for o in n.out_edges):
                    stage_output_ids.append(n.id)

                # discard outgoing edges to external stages
                n.out_edges = [o for o in n.out_edges if o.stage_id == stage_id]

                # add stage inputs
                to_replace = dict()
                for u in n.in_edges:
                    if (u.stage_id != stage_id) or (u.type is NodeTypes.IN):
                        if u.id in stage_inputs:
                            stage_input = stage_inputs[u.id]
                        else:
                            # create a new input node for this stage
                            stage_input = Node.from_other(u)
                            stage_input.type = NodeTypes.IN
                            stage_input.args = []
                            stage_input.kwargs = dict()
                            stage_input.stage_id = stage_id
                            stage_input.out_edges = [o for o in u.out_edges if o.stage_id == stage_id]
                            stage_inputs[u.id] = stage_input
                            stage_nodes[u.id] = stage_input
                        to_replace[u] = stage_input

                    if u.id in self.input_kw_ids:
                        stage_input_kws[u.id] = self.input_kw_ids[u.id]

                # replace inputs
                for old, new in to_replace.items():
                    n.replace_input(old, new)
                    new.add_out_edge(n)
            stages[stage_id] = Graph(stage_nodes, stage_input_kws, stage_output_ids, self.depth, self.basic_blocks)

        return stages

    def _remove_parallel_edges(self) -> "Graph":
        """the control flow graph can contain parallel in/out edges
        those edges are important for control flow but are detrimental for partitioning
        this function creates a new Graph without parallel edges"""

        copy = Graph(None, None, None, None,
                     None).load_state(self.state())

        for n in copy.nodes:
            n.out_edges = set(n.out_edges)
            in_edges = n.in_edges
            n.args = set(in_edges)
            n.kwargs.clear()

            if n in n.out_edges:
                warnings.warn(f"Node with self loop {n}")
                n.out_edges.remove(n)
            if n in n.args:
                warnings.warn(f"Node with self loop {n}")
                n.args.remove(n)

            n.args = sorted(n.args, key=lambda x: x.id)
            n.out_edges = sorted(n.out_edges, key=lambda x: x.id)

        return copy

    def topo_sort(self, verbose=False, change_graph=True, resort_edges=False):

        def print_if_verbose(*args, **kw):
            if verbose:
                print(*args, **kw)

        # To networkx
        G = nx.DiGraph()
        for u in self.nodes:
            dsts = set()
            for v in u.out_edges:
                # disallow parallel in edges
                # disallow parallel out edges
                if v.id in dsts:
                    continue
                dsts.add(v.id)
                G.add_edge(u.id, v.id)

        nx_graph = self.asNetworkx(directed=True)

        # key function, which will result in breaking ties by node id.
        def key(node):
            return None

        topo_sorted = nx.dag.lexicographical_topological_sort(nx_graph, key=key)
        topo_sorted = list(topo_sorted)
        for topo_sort_id, node_id in enumerate(topo_sorted):
            self[node_id].topo_sort_id = topo_sort_id

        # if resort_edges:
        #     for node in self.nodes:
        #         node.out_edges.sort(key=lambda x:x.topo_sort_id)
        #         node.args.sort(key=lambda x: x.topo_sort_id)

        if not change_graph:
            return

        if (ids_sort := sorted(topo_sorted)) != topo_sorted:
            print("-W- topo_sort: node_ids are not topo sorted!")
            print_if_verbose("topo_sorted:", topo_sorted)
            print_if_verbose("node_ids", ids_sort)
            replace_ids_with_topo = True
        else:
            print("-I- topo_sort: node_ids are topo sorted")
            replace_ids_with_topo = False

        if replace_ids_with_topo:
            print("-I- replacing node_ids by topo_sort_ids for graph")

            # Fix nodes
            for node in self.nodes:
                node.id = node.topo_sort_id

            # Fix graph:
            #             self.input_kw_ids = input_kw_ids
            input_kw_ids = {}
            for i, v in self.input_kw_ids.items():
                topo_sort_id = self[i].topo_sort_id
                if topo_sort_id != i:
                    print_if_verbose(f"-I- topo_sort: changed id to input {v}")
                input_kw_ids[topo_sort_id] = v

            self.input_kw_ids = input_kw_ids

            # o
            output_ids = []
            for i in self.output_ids:
                topo_sort_id = self[i].topo_sort_id
                if topo_sort_id != i:
                    print_if_verbose(f"-I- topo_sort: changed id to output {i}")
                output_ids.append(topo_sort_id)
            self.output_ids = output_ids

            # Note this has to be last
            #             self._nodes: GraphNodes = nodes
            _nodes = {n.topo_sort_id: n for n in self.nodes}
            self._nodes = _nodes

    def forward_dfs_and_check(self, source: Node, set_to_check: Set[Node], depth_limit: Optional[int] = None):
        if depth_limit is None:
            depth_limit = len(self)
        # dfs
        nodes = [source]
        visited = set()
        for start in nodes:
            if start in visited:
                continue
            if start in set_to_check:
                return True
            visited.add(start)

            stack = [(start, depth_limit, iter(start.out_edges))]  # : List[Tuple[Node, int, Iterator[Node]]]
            while stack:
                parent, depth_now, children = stack[-1]
                try:
                    child = next(children)
                    if child in visited:
                        continue
                    if child in set_to_check:
                        return True
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(child.out_edges)))

                except StopIteration:
                    stack.pop()
        return False

    def calculate_params_per_node(self, model: Module):
        # Pytorch
        layers = layerDict(model, self.depth, self.basic_blocks)
        tensors = tensorDict(model)

        # params_per_node = dict()

        for n in self.nodes:
            if n.scope in layers:  # n.value_type == NodeTypes.LAYER
                x = sum(t.numel() for t in layers[n.scope].parameters())
            elif (n.value_type is Parameter) and (n.scope in tensors):
                x = tensors[n.scope].numel()
            else:
                x = 0
            n.num_parameters = x

        # return params_per_node

def remove_dups(lnodes: List[Node], myself):
    s = set(lnodes)
    if myself in s:
        s.remove(myself)
    return sorted(s, key=lambda x: x.topo_sort_id)
