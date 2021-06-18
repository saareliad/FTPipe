import warnings
from copy import deepcopy
from typing import List, Optional, Tuple

from sortedcollections import ValueSortedDict

from autopipe.autopipe.model_partitioning.heuristics import EdgeWeightFunction, NodeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.by_prefix import coarsen_prefixes, \
    annotate_special_blocks_to_hold_to
from autopipe.autopipe.model_partitioning.mixed_pipe.centers import stochastic_centers_matching
from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_partitioning.mixed_pipe.systematic_block_ratio_creation import RatioBlockCreator
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node
from autopipe.autopipe.union_find import UnionFind


def coarsening(model, graph,
               edge_weight_function: EdgeWeightFunction,
               node_weight_function: NodeWeightFunction,
               L, P,
               basic_blocks,
               special_blocks,
               depth) -> List[Tuple[Graph, List[List[Node]], Graph, UnionFind]]:
    # uf = UnionFind(elements=[n.id for n in graph.non_input_nodes])
    print(f"-I- Coarsening: got graph with {graph.num_nodes} nodes")

    mgr = CoarseningMgr(model, graph,
                        edge_weight_function,
                        node_weight_function,
                        L, P,
                        basic_blocks,
                        special_blocks,
                        depth)
    mgr.add_method("prefixes")
    mgr.add_method("forbidden_edges")
    mgr.add_method("node_weight_0")
    # mgr.add_method("heavy_edges", 0.99)  # enable this only if there is problems with CCO.
    mgr.add_method("cco")
    mgr.add_method("stochastic_centers")
    mgr.add_method("smallest_nodes")

    mgr.execute()

    return mgr.hierarchy


class CoarseningMgr:
    def __init__(self,
                 model,
                 graph: Graph,
                 edge_weight_function: EdgeWeightFunction,
                 node_weight_function: NodeWeightFunction,
                 L,
                 P,
                 basic_blocks,
                 special_blocks,
                 depth
                 ):

        # load args
        self.model = model
        self.graph = graph
        self.edge_weight_function = edge_weight_function
        self.node_weight_function = node_weight_function
        self.L = L
        self.P = P
        self.basic_blocks = basic_blocks
        self.special_blocks = special_blocks
        self.depth = depth

        print(f"-I- Coarsening: got graph with {graph.num_nodes} nodes")

        # Load internal state
        self.uf = UnionFind(elements=[n.id for n in graph.non_input_nodes])
        self.p = graph
        self.pipeline = []
        self.kwargs_pipeline = []
        self.hierarchy = []

    def add_method(self, name, *method_args, **method_kwargs):
        self.pipeline.append(name)
        self.kwargs_pipeline.append((method_args, method_kwargs))

    def execute(self):
        if "prefixes" in self.pipeline:
            annotate_special_blocks_to_hold_to(model=self.model, graph=self.graph, special_blocks=self.special_blocks,
                                               basic_blocks=self.basic_blocks,
                                               depth=self.depth)

        for i, (method, (method_args, method_kwargs)) in enumerate(zip(self.pipeline, self.kwargs_pipeline)):
            is_last_op_in_pipe = i == len(self.pipeline) - 1

            if method == "prefixes":
                self.coarsen_prefixes(is_last_op_in_pipe=is_last_op_in_pipe)
            elif method == "forbidden_edges":
                self.forbidden_edges(is_last_op_in_pipe=is_last_op_in_pipe)
            elif method == "node_weight_0":
                self.node_weight_0(is_last_op_in_pipe=is_last_op_in_pipe)
            elif method == "heavy_edges":
                self.heavy_edges(*method_args, **method_kwargs, is_last_op_in_pipe=is_last_op_in_pipe)
            elif method == "cco":
                self.cco(is_last_op_in_pipe=is_last_op_in_pipe)
            elif method == "stochastic_centers":
                self.stochastic_centers(is_last_op_in_pipe=is_last_op_in_pipe)
            elif method == "smallest_nodes":
                self.smallest_nodes(is_last_op_in_pipe=is_last_op_in_pipe)
            else:
                raise NotImplementedError(method)

    def append_to_hierarchy(self, p, uf2, g, uf, is_last_op_in_pipe=False):
        hierarchy = self.hierarchy
        if not is_last_op_in_pipe:
            hierarchy.append((p, uf2, g, deepcopy(uf)))
        else:
            hierarchy.append((p, uf2, g, uf))

    def coarsen_prefixes(self, is_last_op_in_pipe=False):
        p, _, g, uf, uf2, sb_names = coarsen_prefixes(model=self.model,
                                                      graph=self.p,
                                                      node_weight_function=self.node_weight_function,
                                                      edge_weight_function=self.edge_weight_function,
                                                      uf=self.uf,
                                                      basic_blocks=self.basic_blocks,
                                                      special_blocks=self.special_blocks,
                                                      depth=self.depth)
        print(sb_names)
        print(f"merged {len(p) - len(g)} nodes with common prefixes")

        self.append_to_hierarchy(p, uf2, g, uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g

    def forbidden_edges(self, is_last_op_in_pipe=False):
        matching = penalty_edges_matching(graph=self.p, edge_weight_function=self.edge_weight_function)
        g = contract(self.p, matching, self.edge_weight_function, uf=self.uf)
        print(f"merged {len(matching)} nodes with penalty edges")

        # FIXME: matching arg should instead uf2.
        self.append_to_hierarchy(self.p, matching, g, self.uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g

    def node_weight_0(self, is_last_op_in_pipe=False):
        print(f"merging nodes with weight <=0")
        p, _, g, uf, uf2 = nodes_leq_threshold_matching(self.p,
                                                        self.node_weight_function,
                                                        self.edge_weight_function,
                                                        self.L,
                                                        self.uf,
                                                        verbose=False,
                                                        record_history=True,
                                                        threshold=0)
        self.append_to_hierarchy(p, uf2, g, self.uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g

    def heavy_edges(self, percentile_to_filter=0.95, is_last_op_in_pipe=False):
        p, _, g, uf, uf2 = online_heavy_edge_matching(self.p,
                                                      self.node_weight_function,
                                                      self.edge_weight_function,
                                                      self.L,
                                                      self.uf,
                                                      verbose=True,
                                                      record_history=True,
                                                      pecentile_to_filter=percentile_to_filter)

        self.append_to_hierarchy(p, uf2, g, self.uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g

    def cco(self, is_last_op_in_pipe=False):
        # systematic
        p, _, g, uf, uf2 = systematic_comm_comp_ratio_matching(
            self.p,
            self.node_weight_function,
            self.edge_weight_function,
            self.L,
            self.uf,
            verbose=True,
        )
        if uf2 is None:
            warnings.warn("can't restore single step of systematic max blocks")

        self.append_to_hierarchy(p, uf2, g, self.uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g

    def stochastic_centers(self, is_last_op_in_pipe=False):
        p, _, g, uf, uf2 = stochastic_centers_matching(self.p, self.node_weight_function,
                                                       self.edge_weight_function,
                                                       self.L, self.P, self.uf, verbose=True, record_history=False,
                                                       special_blocks=self.special_blocks)

        self.append_to_hierarchy(p, uf2, g, self.uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g

    def smallest_nodes(self, is_last_op_in_pipe=False):
        p, _, g, uf, uf2 = online_smallest_comp_node_matching(self.p,
                                                              self.node_weight_function,
                                                              self.edge_weight_function,
                                                              self.L,
                                                              self.uf,
                                                              verbose=True,
                                                              record_history=True
                                                              )
        self.append_to_hierarchy(p, uf2, g, self.uf, is_last_op_in_pipe=is_last_op_in_pipe)
        self.p = g


def contract(graph: Graph, matching: List[List[Node]], edge_weight_function: EdgeWeightFunction,
             uf: Optional[UnionFind] = None) -> Graph:
    # if not matching:
    #     return graph
    new_graph = Graph.from_other(graph)
    # Start from end, so when we merge outputs are already handled
    for m in sorted(matching, key=lambda x: x[0].id, reverse=True):
        root = m[0]
        for i in m[1:]:
            new_graph.merge(root.id, i.id, edge_weight_function=edge_weight_function, uf=uf)
            if uf is not None:
                uf.union(x=root.id, y=i.id)
    return new_graph


def penalty_edges_matching(graph: Graph, edge_weight_function: EdgeWeightFunction):
    """Penalized edges are for disallowing sending weird stuff which MPI and the like can't handle.
        # TODO: if this creates a cycle we have nothing to do, but manually wrap it and disallow communication of weird stuff
    """
    matching = []
    for node in graph.non_input_nodes:
        check = False
        for out in node.out_edges:
            if edge_weight_function(node, out) >= edge_weight_function.penalty:
                if check_cycle2(graph, node, out):
                    warnings.warn(f"can't compress edge with penalty (node,out)={(node, out)}")
                    continue
                matching.append([node, out])  # <---- into node
                # TODO: we have to handle doubles and so on...
                check = True
        if check:
            if not node.compound_edge_weights:

                try:
                    for out in node.out_edges:
                        assert edge_weight_function(node, out) >= edge_weight_function.penalty
                except AssertionError:
                    for out in node.out_edges:
                        print(edge_weight_function(node, out))
                    print("PENATLY", edge_weight_function.penalty)

                count=0
                for v in node.out_edges:
                    if v.id in graph.output_ids:
                        continue
                    count += 1
                for e in matching[-count:]:
                    if e[0] is not node:
                        # This should happen, since penalty is on the node itself
                        print("matching")
                        for tup in matching:
                            print(tup)

                        print("out edges")
                        for v in node.out_edges:
                            print(v)

                        raise NotImplementedError(
                            f"potential cycle in edge {e}. (count={count}) Should probably duplicate node, or check topo order.")
    return matching


def code_analysis_matching(graph: Graph):
    pass


def adjacent_and_same_size_matching(graph: Graph):
    pass


def systematic_comm_comp_ratio_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                                        verbose=False):
    prev_graph = Graph.from_other(graph)

    rbc = RatioBlockCreator(graph, edge_weight_function=edge_weight_function, node_weight_function=node_weight_function,
                            uf=uf)
    rbc.apply(L, verbose=verbose)

    matching = None
    return prev_graph, matching, graph, uf, None


def online_smallest_comp_node_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                                       verbose=False, record_history=False):
    prev_graph = Graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())

    hd = ValueSortedDict({
        n: node_weight_function(n) for n in graph.non_input_nodes
    })

    def inner_loop():
        # optimization: can use the index of new item to skip initial checks if there is no match in them.
        # But it works good enough without it.
        for u, weight_of_u in hd.items():
            # Try to find match:
            for v in sorted(u.out_edges, key=lambda n: node_weight_function(n)):
                if check_cycle2(graph, u, v):
                    # can't merge without breaking topo sort
                    continue
                graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
                uf.union(u.id, v.id)
                uf2.union(u.id, v.id)
                hd.pop(u)
                hd.pop(v)
                hd[u] = node_weight_function(u)
                return True, weight_of_u
        return False, None

    history_sizes = []
    history_weights = []
    while len(hd) > L:
        # u, weight_of_u = hd.peekitem()
        merged_something, weight_of_u = inner_loop()
        if not merged_something:
            break
        if record_history:
            history_sizes.append(len(hd) + 1)
            history_weights.append(weight_of_u)
        if verbose:
            print(f"Nodes: {len(hd)}, Smallest: {weight_of_u}")

    # Note: matching is pretty much meaningless.
    matching = None
    return prev_graph, matching, graph, uf, uf2


def nodes_leq_threshold_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                                 verbose=False, record_history=False, threshold=0):
    prev_graph = Graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())

    hd = ValueSortedDict({
        n: node_weight_function(n) for n in graph.non_input_nodes
    })
    total_merged = 0

    def inner_loop():
        # optimization: can use the index of new item to skip initial checks if there is no match in them.
        # But it works good enough without it.
        for u, weight_of_u in hd.items():
            if weight_of_u > threshold:
                print(
                    f"done with  nodes <= threshold {threshold}, breaking (last weight: {weight_of_u}). merged {total_merged}")
                return False, None, True

            u: Node
            for v in sorted(u.in_edges, key=lambda n: node_weight_function(n)):
                if v in graph.inputs:
                    continue
                if check_cycle2(graph, v, u):
                    # can't merge without breaking topo sort
                    continue
                graph.merge(uid=v.id, vid=u.id, edge_weight_function=edge_weight_function, uf=uf)
                uf.union(v.id, u.id)
                uf2.union(v.id, u.id)
                hd.pop(v)
                hd.pop(u)
                hd[v] = node_weight_function(v)
                return True, weight_of_u, False

            # Try to find match, forward
            for v in sorted(u.out_edges, key=lambda n: node_weight_function(n)):
                if check_cycle2(graph, u, v):
                    # can't merge without breaking topo sort
                    continue
                warnings.warn(f"can't merge small node {u} backward, will merge forward and lose the name of {v}.")
                # if "T5Block" in v.scope:
                #     print("HERE")
                graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
                uf.union(u.id, v.id)
                uf2.union(u.id, v.id)
                hd.pop(u)
                hd.pop(v)
                hd[u] = node_weight_function(u)
                return True, weight_of_u, False
        return False, None, False

    history_sizes = []
    history_weights = []
    while len(hd) > L:
        # u, weight_of_u = hd.peekitem()
        merged_something, weight_of_u, threshold_cond = inner_loop()
        if threshold_cond:
            break
        if not merged_something:
            break

        total_merged += 1
        if record_history:
            history_sizes.append(len(hd) + 1)
            history_weights.append(weight_of_u)
        if verbose:
            print(f"Nodes: {len(hd)}, Smallest: {weight_of_u}")

    # Note: matching is pretty much meaningless.
    matching = None
    return prev_graph, matching, graph, uf, uf2


def ofline_smallest_comp_node_matching(graph: Graph, node_weight_function):
    matching = []
    matched = set()

    for u in sorted(graph.non_input_nodes, key=lambda n: node_weight_function(n)):
        # Try to find match:
        if u in matched:
            continue
        for v in sorted(u.out_edges, key=lambda n: node_weight_function(n)):
            if v in matched:
                continue
            if check_cycle2(graph, u, v):
                # can't merge without breaking topo sort
                continue
            matched.add(u)
            matched.add(v)
            matching.append((u, v))
    return matching


def online_heavy_edge_matching(graph: Graph, node_weight_function, edge_weight_function, L, uf: UnionFind,
                               verbose=False, record_history=False, pecentile_to_filter=0.9):
    prev_graph = Graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())
    # HACK: re-using code from RatioBlockCreator
    rbc = RatioBlockCreator(graph, edge_weight_function=edge_weight_function, node_weight_function=node_weight_function,
                            uf=uf)

    hd = rbc.sorted_graph_forward_edges(descending=True)  # ValueSortedDict

    def inner_loop():
        # optimization: can use the index of new item to skip initial checks if there is no match in them.
        # But it works good enough without it.
        for (uid, vid), weight_of_u_v in hd.items():
            u = graph[uid]
            v = graph[vid]

            # Try to find match:
            if check_cycle2(graph, u, v):
                # can't merge without breaking topo sort
                continue
            graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
            # uf.union(u.id, v.id) handled below
            uf2.union(u.id, v.id)
            rbc.update_sorted_edges_on_merges(edges_to_value=hd, merges=[(u.id, v.id)], allow_poped_outside=True)
            return True, weight_of_u_v
        return False, None

    history_sizes = []
    history_weights = []
    import pandas as pd
    s = pd.Series(list(hd.values()))
    description = s.describe(percentiles=[0.5, 0.75, 0.8, 0.9, 0.95, 0.99])
    print(description)

    if pecentile_to_filter is not None:
        dest_length = len(hd) * pecentile_to_filter
        print(f"Filtering hte {pecentile_to_filter} percentile")
    else:
        dest_length = L

    while len(hd) > dest_length:
        # u, weight_of_u = hd.peekitem()
        merged_something, weight_of_merged = inner_loop()
        if not merged_something:
            break
        if record_history:
            history_sizes.append(len(hd) + 1)
            history_weights.append(weight_of_merged)
        if verbose:
            print(f"Edges: {len(hd)}, Largest edge: {weight_of_merged}")

    # Note: matching is pretty much meaningless.
    matching = None
    return prev_graph, matching, graph, uf, uf2
