from collections import defaultdict
from pytorch_Gpipe.model_profiling import Node, Graph
from typing import Dict, Iterable, Set, Iterator, List, Tuple, Any, Optional
import random
import heapq
import pickle
from collections.abc import MutableMapping

###################################################################################################

# The purpose of the following two classes is twofold:
# (1) Provide dyanmic weights
# (2) to be transparnt to the algorithm as mucha s possible


class DynamicNodeWeights(MutableMapping):
    def __init__(self, node_weights, node_weight_function):

        self.store = node_weights
        # self.work_graph = work_graph
        self.node_weight_function = node_weight_function

    @classmethod
    def from_graph(cls, work_graph, node_weight_function):
        node_weights = dict()
        # Full init
        for n in work_graph.nodes:
            node_weights[n] = node_weight_function(n)
        return cls(node_weights, node_weight_function)

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def recalculate_weight(self, key):
        self.store[key] = self.node_weight_function(key)


class DynamicEdgeWeights(MutableMapping):
    def __init__(self,
                 edge_weights,
                 edge_weight_function,
                 backward_edged=True):
        # TODO: will provide option to turn off backward_edged

        backward_edges = set(
            (edge[1], edge[0]) for edge in edge_weights.keys())
        for bwd_edge in backward_edges:
            edge_weights[bwd_edge] = edge_weight_function(*bwd_edge)

        self.store = edge_weights
        # self.work_graph = work_graph
        self.edge_weight_function = edge_weight_function
        self._backward_edges = backward_edges

    @classmethod
    def from_graph(cls, work_graph, edge_weight_function):
        edge_weights = dict()

        for n in work_graph.nodes:
            for o in n.out_edges:
                # Forward (normal) edges
                edge_weights[(n, o)] = edge_weight_function(n, o)

        return cls(edge_weights, edge_weight_function)

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def recalculate_weight(self, key):
        # NOTE: user should call twich in directed case
        self.store[key] = self.edge_weight_function(key[0], key[1])

    def clear_backward_edges(self):
        """ backward edges are not part of the original graph,
            can be used just for (directed) heuristics.
        """
        for key in self._backward_edges:
            del self.store[key]


class StaticNodeWeights(DynamicNodeWeights):
    def __init__(self, *args, **kw):
        super().__init__()

    def recalculate_weight(self, key):
        pass


class StaticEdgeWeights(DynamicEdgeWeights):
    def __init__(self, *args, **kw):
        super().__init__()

    def recalculate_weight(self, key):
        pass


###################################################################################################


class DoublePriority():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __neg__(self):
        return DoublePriority(-self.a, -self.b)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.a < other
        return (self.a < other.a) or ((self.a == other.a) and
                                      (self.b < other.b))

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.a > other
        return (self.a > other.a) or ((self.a == other.a) and
                                      (self.b > other.b))

    def __isub__(self, other):
        self.a -= other.a
        self.b -= other.b
        return self

    def __le__(self, other):
        return not (self > other)

    def __str__(self):
        return f"(a:{self.a},b:{self.b})"

    def __repr__(self):
        return str(self)


class PriorityQueue():
    #heapq is a min heap and we need a max heap
    #so we push the negative gain
    #we use a random tie braker in case several tasks have the same gain
    def __init__(self):
        self.heap = []

    def push_task(self, gain: float, task: Any):
        tie_braker = random.randint(0, 2**32)
        priority = (-gain, -tie_braker)

        heapq.heappush(self.heap, (priority, task))

    def pop_task(self) -> Any:
        priority, task = heapq.heappop(self.heap)
        return task

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return len(self) > 0


class PartitionNode():
    """ PartitionNode is a collection of graph nodes allocated to the same partition
        an edge exists between PartitionNodes iff they there are edges between the underlying graph nodes
    """
    def __init__(self, nodes: Iterable[Node], idx: int):
        self.nodes: Set[Node] = set(nodes)
        self._out_edges = defaultdict(lambda: 0)
        self._in_edges = defaultdict(lambda: 0)
        self.id = idx

        for n in self.nodes:
            for i in n.in_edges:
                self._in_edges[i.stage_id] += 1
            for o in n.out_edges:
                self._out_edges[o.stage_id] += 1

        self._out_edges.pop(self.id, None)
        self._in_edges.pop(self.id, None)

    @property
    def in_edges(self) -> List[int]:
        return [i for i, n in self._in_edges.items() if n > 0]

    @property
    def out_edges(self) -> List[int]:
        return [i for i, n in self._out_edges.items() if n > 0]

    def __contains__(self, key) -> bool:
        return key in self.nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def add_in_edge(self, src: int):
        self._in_edges[src] += 1

    def add_out_edge(self, dst: int):
        self._out_edges[dst] += 1

    def remove_in_edge(self, src: int):
        self._in_edges[src] -= 1

    def remove_out_edge(self, dst: int):
        self._out_edges[dst] -= 1

    def add_node(self, node: Node):
        self.nodes.add(node)

    def remove_node(self, node: Node):
        self.nodes.discard(node)


class QuotientGraph():
    def __init__(self, nodes: Iterable[Node]):
        groups = defaultdict(list)
        for n in nodes:
            groups[n.stage_id].append(n)

        self._nodes: Dict[int, PartitionNode] = {
            idx: PartitionNode(group, idx)
            for idx, group in groups.items()
        }

    def __getitem__(self, idx: int) -> PartitionNode:
        return self._nodes[idx]

    def move_node(self, node: Node, dst: int):
        assert node.stage_id != dst
        src = node.stage_id
        src_part = self[src]
        dst_part = self[dst]

        src_part.remove_node(node)
        dst_part.add_node(node)
        node.stage_id = dst

        for i in node.in_edges:
            i_part = self[i.stage_id]
            #remove edge from i to src
            src_part.remove_in_edge(i.stage_id)
            i_part.remove_out_edge(src)

            #add edge from i to dest
            i_part.add_out_edge(dst)
            dst_part.add_in_edge(i.stage_id)

        for o in node.out_edges:
            o_part = self[o.stage_id]
            #remove edge from src to o
            src_part.remove_out_edge(o.stage_id)
            o_part.remove_in_edge(src)

            #add edge from dst to o
            o_part.add_in_edge(dst)
            self[dst].add_out_edge(o.stage_id)

        #remove self edges
        #faster than using if statements in the for loops
        for p in self.nodes:
            p._in_edges.pop(p.id, None)
            p._out_edges.pop(p.id, None)

    def move_creates_cycle(self, node: Node, dest: int) -> bool:
        orig_part = node.stage_id
        self.move_node(node, dest)
        creates_cycle = self.has_cycles()
        self.move_node(node, orig_part)
        return creates_cycle

    @property
    def nodes(self) -> Iterable[PartitionNode]:
        return self._nodes.values()

    def has_cycles(self) -> bool:
        S = []
        T = []
        degs = dict()
        # o(V)
        for n in self.nodes:
            assert isinstance(n, PartitionNode)
            if len(n.in_edges) == 0:
                S.append(n.id)
            else:
                degs[n] = len(n.in_edges)

        # O(E)
        while S:
            n = self._nodes[S.pop()]
            assert isinstance(n, PartitionNode)
            T.append(n)
            for o in n.out_edges:
                out = self._nodes[o]
                assert isinstance(out, PartitionNode)
                degs[out] -= 1
                if degs[out] == 0:
                    S.append(o)

        #if we have a cycle S will be empty and T will not contain all of the nodes
        return len(T) < len(self.nodes)

    def build_dot(self):
        '''
        return a graphviz representation of the graph
        Parameters
        ----------
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
            dot.node(str(node.id),
                     label=f"partition:{node.id}",
                     fillcolor=colors[node.id])
            for i in node.in_edges:
                dot.edge(str(i), str(node.id))

        return dot

    def save_as_pdf(self, file_name: str, directory: str):
        '''
        save the rendered graph to a pdf file

        Parameters
        ----------
        file_name:
            the name of the saved file
        directory:
            directory to store the file in
        '''
        dot = self.build_dot()
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)
        return self

    def print_stats(self, node_weights: Dict[Node, float],
                    edge_weights: Dict[Tuple[Node, Node], float]):
        volumes = defaultdict(lambda: 0)
        edge_cut = 0
        number_of_cutting_edges = 0
        for partition in self.nodes:
            for n in partition:
                volumes[partition.id] += node_weights[n]
                for o in n.out_edges:
                    if n.stage_id != o.stage_id:
                        if edge_weights[(n, o)] >= 1000:
                            print(f"{n.id}=>{o.id}")
                            print(f"{n.stage_id}=>{o.stage_id}")
                            print(f"{n.value_type}")
                            print(f"weight:{edge_weights[(n,o)]:.2f}\n")
                        edge_cut += edge_weights[(n, o)]
                        number_of_cutting_edges += 1

        total_volume = sum(volumes.values())
        avg_volume = total_volume / len(volumes)
        print(f"total number of nodes: {len(node_weights)}")
        print(f"total number of edges: {len(edge_weights)}")
        print(f"total weight: {total_volume:.2f}")
        print(f"avg weight: {avg_volume:.2f}")
        print(f"number of cutting edges: {number_of_cutting_edges}")
        print(f"edge cut: {edge_cut:.2f}")
        print("partition stats:")
        for i in range(len(volumes)):
            print(f"    partition {i}")
            print(f"    number of nodes {len(self._nodes[i])}")
            print(f"    partition volume: {volumes[i]:.2f}\n")

    def selfcheck(self):
        visited = set()
        for idx, n in self._nodes.items():
            assert idx == n.id
            for u in n.nodes:
                assert u.stage_id == idx
                assert u not in visited
                visited.add(u)

            for i, v in n._in_edges.items():
                assert v >= 0, (idx, i, v)
            assert idx not in n._in_edges
            for i in n.in_edges:
                assert idx in self._nodes[i].out_edges, (idx, i)

            for o, v in n._out_edges.items():
                assert v >= 0, (idx, o, v)
            assert idx not in n._out_edges
            for o in n.out_edges:
                assert idx in self._nodes[o].in_edges, (idx, o)

        assert not self.has_cycles()


class VerticeStageConnections():
    def __init__(self, nodes):
        self._in_connections = dict()
        self._out_connections = dict()

        for n in nodes:
            self._in_connections[n] = defaultdict(lambda: 0)
            self._out_connections[n] = defaultdict(lambda: 0)

        for n in nodes:
            for u in n.in_edges:
                self._in_connections[n][u.stage_id] += 1
                self._out_connections[u][n.stage_id] += 1

    def add_in_connection(self, n, src: int):
        self._in_connections[n][src] += 1

    def add_out_connection(self, n, dest: int):
        self._out_connections[n][dest] += 1

    def remove_in_connection(self, n, src: int):
        self._in_connections[n][src] -= 1

    def remove_out_connection(self, n, dest: int):
        self._out_connections[n][dest] -= 1

    def has_in_connection(self, n, src: int) -> bool:
        return self._in_connections[n][src] > 0

    def has_out_connection(self, n, dest: int) -> bool:
        return self._out_connections[n][dest] > 0

    def in_connections(self, n, src: int) -> int:
        return self._in_connections[n][src]

    def out_connections(self, n, dst: int) -> int:
        return self._out_connections[n][dst]

    def move_node(self, n, src: int, dest: int):
        for u in n.in_edges:
            self.remove_out_connection(u, src)
            self.add_out_connection(u, dest)
        for o in n.out_edges:
            self.remove_in_connection(o, src)
            self.add_in_connection(o, dest)


class Path():
    def __init__(self, v):
        self.start = self.end = v
        self.length = 0
        self.active = True

    def is_cycle(self) -> bool:
        return (self.start is self.end) and (self.length > 0)


class PathSet():
    def __init__(self, graph_nodes: Iterable[Node]):
        self.paths = {v: Path(v) for v in graph_nodes}

        self.next: Dict[Node, Node] = {v: v for v in graph_nodes}
        self.prev: Dict[Node, Node] = {v: v for v in graph_nodes}

        self.next_edge: Dict[Node, Optional[Tuple[Node, Node]]] = {
            v: None
            for v in graph_nodes
        }
        self.prev_edge: Dict[Node, Optional[Tuple[Node, Node]]] = {
            v: None
            for v in graph_nodes
        }

        self.n_active_paths = len(self.paths)

    def is_endpoint(self, v: Node) -> bool:
        return (self.next[v] is v) or (self.prev[v] is v)

    def next_vertex(self, v: Node) -> Node:
        return self.next[v]

    def prev_vertex(self, v: Node) -> Node:
        return self.prev[v]

    def edge_to_next(self, v: Node) -> Optional[Tuple[Node, Node]]:
        return self.next_edge[v]

    def edge_to_prev(self, v: Node) -> Optional[Tuple[Node, Node]]:
        return self.prev_edge[v]

    def add_if_eligible(self, edge: Tuple[Node, Node]) -> bool:
        src, dst = edge

        src_path = self.paths[src]
        dst_path = self.paths[dst]

        assert src is not dst

        #edges between partitions are not eligible
        if src.stage_id != dst.stage_id:
            return False

        # both vertices must be endpoints in order for the edge to be eligible
        if not (self.is_endpoint(src) and self.is_endpoint(dst)):
            return False

        assert src_path.active and dst_path.active

        # edge to/from cycle is not eligible
        if (src_path.is_cycle() or dst_path.is_cycle()):
            return False

        if src_path is not dst_path:
            #we do not close a cycle so we merge paths
            self.n_active_paths -= 1
            src_path.length += (dst_path.length + 1)

            #update paths basically handle the 4 possible direction combinations
            if (src_path.start is src and dst_path.start is dst):
                self.paths[dst_path.end] = src_path
                src_path.start = dst_path.end
            elif (src_path.start is src and dst_path.end is dst):
                self.paths[dst_path.start] = src_path
                src_path.start = dst_path.start
            elif (src_path.end is src and dst_path.start is dst):
                self.paths[dst_path.end] = src_path
                src_path.end = dst_path.end
            elif (src_path.end is src and dst_path.end is dst):
                self.paths[dst_path.start] = src_path
                src_path.end = dst_path.start

            #update the doubly linked list
            if self.next[src] is src:
                assert self.next_edge[src] is None
                self.next[src] = dst
                self.next_edge[src] = edge
            else:
                assert self.prev_edge[src] is None
                self.prev[src] = dst
                self.prev_edge[src] = edge

            if self.next[dst] is dst:
                assert self.next_edge[dst] is None
                self.next[dst] = src
                self.next_edge[dst] = edge
            else:
                assert self.prev_edge[dst] is None
                self.prev[dst] = src
                self.prev_edge[dst] = edge

            #deactivate the path as it has been merged
            dst_path.active = False

        elif (src_path.length % 2) == 1:
            # close even length cycle
            src_path.length += 1

            # close the cycle by updateing the doubly linked list
            if self.next[src_path.start] is src_path.start:
                self.next[src_path.start] = src_path.end
                self.next_edge[src_path.start] = edge
            else:
                self.prev[src_path.start] = src_path.end
                self.prev_edge[src_path.start] = edge

            if self.next[src_path.end] is src_path.end:
                self.next[src_path.end] = src_path.start
                self.next_edge[src_path.end] = edge
            else:
                self.prev[src_path.end] = src_path.start
                self.prev_edge[src_path.end] = edge

            src_path.end = src_path.start
            return True

        return False

    def active_paths(self) -> Set[Path]:
        paths = [p for p in self.paths.values() if p.active]
        return set(paths)


# TODO: add weight somehow
class SimpleNode():
    def __init__(self, idx, stage_id):
        self.id = idx
        self.in_edges = set()
        self.out_edges = set()
        self.stage_id = stage_id

    def add_in_edge(self, node):
        self.in_edges.add(node)

    def add_out_edge(self, node):
        self.out_edges.add(node)


# TODO: use this to make the update support dynamic weights.
class ContractedGraphDynamicNodeWeights(DynamicNodeWeights):
    def __init__(self, contracted_graph: "ContractedGraph", *args, **kw):
        super().__init__(*args, **kw)
        self.contracted_graph = contracted_graph

    def recalculate_weight(self, key):
        # TODO: get the head node
        # TODO: update weight of the current node
        # TODO: update weight of that head node.
        self.store[key] = self.node_weight_function(key)
        # Will raise


# TODO: make it work with DynamicNodeWeights.
class ContractedGraph():
    def __init__(
        self,
        in_edges,
        partition,
        node_weights,
        edge_weights,
        matching,
        node_weight_function,
        edge_weight_function,
    ):
        self._nodes: Dict[int, SimpleNode] = dict()
        # FIXME: to support DynamicNodeWeights, something must be changed.
        # currently, the ommited wieght does not let us use DynamicNodeWeights
        for n in set(matching.values()):
            self._nodes[n] = SimpleNode(n, partition[n])

        self._node_weights = defaultdict(lambda: 0)
        self._edge_weights = defaultdict(lambda: 0)

        for n in node_weights.keys():
            matched = matching[n]
            self._node_weights[self._nodes[matched]] += node_weights[n]
            for i in in_edges[n]:
                matched_i = matching[i]
                if matched_i == matched:
                    continue
                self._nodes[matched].add_in_edge(self._nodes[matched_i])
                self._nodes[matched_i].add_out_edge(self._nodes[matched])

                self._edge_weights[(self._nodes[matched_i],
                                    self._nodes[matched])] += edge_weights[(i,
                                                                            n)]
        self._node_weights = ContractedGraphDynamicNodeWeights(
            self, self._node_weights, node_weight_function)
        self._edge_weights = ContractedGraphDynamicNodeWeights(
            self, self._edge_weights, edge_weight_function)

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, idx) -> SimpleNode:
        return self._nodes[idx]

    def node_weight(self, n) -> float:
        return self._node_weights[n]

    def edge_weight(self, u, v) -> float:
        return self._edge_weights[u, v]

    @property
    def nodes(self) -> Iterable[SimpleNode]:
        return self._nodes.values()

    def selfcheck(self) -> "ContractedGraph":
        for idx, n in self._nodes.items():
            assert n.id == idx
            assert n in self._node_weights
            for u in n.in_edges:
                # assert n.id > u.id
                assert n.stage_id >= u.stage_id
                assert n in u.out_edges
                assert (u, n) in self._edge_weights
                assert u in self._node_weights
                assert u.id in self._nodes

            for o in n.out_edges:
                # assert n.id < o.id
                assert o.stage_id >= n.stage_id
                assert n in o.in_edges
                assert (n, o) in self._edge_weights
                assert o in self._node_weights
                assert o.id in self._nodes

        return self

    @classmethod
    def contract(cls, contracted_graph, matching, *args,
                 **kw) -> "ContractedGraph":
        in_edges = dict()
        partition = dict()
        node_weights = dict()
        edge_weights = dict()

        for n in contracted_graph.nodes:
            node_weights[n.id] = contracted_graph.node_weight(n)
            partition[n.id] = n.stage_id
            us = set()
            for u in n.in_edges:
                us.add(u.id)
                edge_weights[(u.id, n.id)] = contracted_graph.edge_weight(u, n)
            in_edges[n.id] = us

        return cls(in_edges, partition, node_weights, edge_weights, matching,
                   *args, **kw)

    @classmethod
    def from_Graph(cls, graph: Graph, node_weights, edge_weights, *args,
                   **kw) -> "ContractedGraph":
        node_weights = {n.id: w for n, w in node_weights.items()}
        edge_weights = {(u.id, v.id): w for (u, v), w in edge_weights.items()}
        in_edges = dict()
        partition = dict()
        for n in graph.nodes:
            in_edges[n.id] = {u.id for u in n.in_edges}
            partition[n.id] = n.stage_id

        matching = {n: n for n in node_weights}
        return cls(in_edges, partition, node_weights, edge_weights, matching,
                   *args, **kw)

    def build_dot(self):
        '''
        return a graphviz representation of the graph
        Parameters
        ----------
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
        for node in self._nodes.values():
            dot.node(str(node.id),
                     label=f"Node:{node.id}\nweight:{self.node_weight(node)}",
                     fillcolor=colors[node.stage_id])
            for i in node.in_edges:
                dot.edge(str(i.id),
                         str(node.id),
                         label=f"weight:{self.edge_weight(i,node)}")

        return dot

    def save_as_pdf(self, file_name: str, directory: str):
        '''
        save the rendered graph to a pdf file

        Parameters
        ----------
        file_name:
            the name of the saved file
        directory:
            directory to store the file in
        '''
        dot = self.build_dot()
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)
        return self

    def serialize(self, path):
        edge_weights = dict()
        node_weights = dict()
        partition = dict()
        in_edges = dict()
        for n in self.nodes:
            in_edges[n.id] = [u.id for u in n.in_edges]
            idx = n.id
            partition[idx] = n.stage_id
            node_weights[idx] = self.node_weight(n)
            for i in n.in_edges:
                edge_weights[(i.id, idx)] = self.edge_weight(i, n)

        state = dict(in_edges=in_edges,
                     partition=partition,
                     edge_weights=edge_weights,
                     node_weights=node_weights)

        if not path.endswith(".graph"):
            path += ".graph"

        pickle.dump(state, open(path, "wb"))

    @classmethod
    def deserialize(cls, path):
        if not path.endswith(".graph"):
            path += ".graph"

        state = pickle.load(open(path, "rb"))
        in_edges = state['in_edges']
        partition = state['partition']
        node_weights = state['node_weights']
        edge_weights = state['edge_weights']
        matching = {n: n for n in node_weights.keys()}

        return cls(in_edges, partition, node_weights, edge_weights,
                   matching).selfcheck()
