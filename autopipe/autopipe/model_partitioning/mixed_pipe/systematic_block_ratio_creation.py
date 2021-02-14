import warnings
from collections import deque
from typing import Set, Dict, Tuple, List, Union

from sortedcollections import ValueSortedDict

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction, \
    CoarsenedWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node, NodeTypes
from autopipe.autopipe.union_find import UnionFind


# TODO: look at cuts instead of edges
# TODO: take statisic of "handled nodes".

class RatioBlockCreator:
    def __init__(self, graph: Graph, edge_weight_function: EdgeWeightFunction,
                 node_weight_function: NodeWeightFunction, uf: UnionFind):
        self.graph = graph
        self.ewf = edge_weight_function
        self.nwf = node_weight_function
        self.cwf = CoarsenedWeightFunction(edge_weight_function=edge_weight_function,
                                           node_weight_function=node_weight_function)
        self.uf = uf

        # edges we already handled and don't want to touch
        # self.merged_nodes = set()  # merge(u,v) <-> v in merged_nodes
        self.protected_edges: Set[int, int] = set()
        self.protected_nodes: Set[int] = set()

    def change_protected_before_merge(self, a: Node, b: Node):

        changes_list_out = []
        changes_list_in = []

        for x in b.out_edges:
            if (b.id, x.id) in self.protected_edges:
                changes_list_out.append((b.id, x.id))
        for x in b.in_edges:
            if (x.id, b.id) in self.protected_edges:
                changes_list_in.append((x.id, b.id))

        for edge in changes_list_out:
            if edge in self.protected_edges:
                self.protected_edges.remove(edge)
                self.protected_edges.add((a.id, edge[1]))
        for edge in changes_list_in:
            if edge in self.protected_edges:
                self.protected_edges.remove(edge)
                self.protected_edges.add((edge[0], a.id))

        protected_changed = len(changes_list_in) > 0 or len(changes_list_out) > 0
        return protected_changed

    def apply(self, L, verbose=False):
        # TODO Get initial state of "protected" nodes
        uf = self.uf
        node_to_cuts = self.sorted_block_to_cuts(forward=True, descending=False)
        node_to_ok_fwd = {
            node.id: not self.cwf.is_comm_bounded_forward(node) for node in self.graph.non_input_nodes
        }
        node_to_ok_bwd = {
            node.id: not self.cwf.is_comm_bounded_backward(node) for node in self.graph.non_input_nodes
        }

        node_to_ok_both = {
            x: a and b for x, a, b in zip(node_to_ok_fwd.keys(), node_to_ok_fwd.values(), node_to_ok_bwd.values())
        }

        # TODO: also handle updating the sort, removing merged edges from the graph
        # Needs some union find.
        n_merges = 0
        n_iter = 0

        failed: Set[int] = set()
        failed_then_merged = set()
        n_ok_fwd = sum(1 for x in node_to_ok_fwd.values() if x)
        n_ok_bwd = sum(1 for x in node_to_ok_bwd.values() if x)
        n_ok_both = sum(1 for x in node_to_ok_both.values() if x)

        def print_state():
            n_nodes = len(self.graph)

            n_failed = len(failed)

            d = dict(iter=n_iter, nodes=n_nodes, merges=n_merges, failed=n_failed, ok_fwd=n_ok_fwd, ok_bwd=n_ok_bwd,
                     ok_both=n_ok_both, remaining=len(node_to_cuts))
            print(d)

        print_state()
        print("Handling nodes without merges nodes")

        for n, is_ok in node_to_ok_fwd.items():
            if is_ok:
                node_to_cuts.pop(n, default=None)
                self.protected_nodes.add(n)

        print_state()

        while node_to_cuts and len(self.graph) > L:
            n_iter += 1
            node_id, node_fwd_cut = node_to_cuts.popitem()
            root_left = self.graph[node_id]

            saved_state = self.graph.state()  # O(N). we are dead...
            (is_success, graph_changed, merges_left, protected_node) = self.search_left(root_left,
                                                                                        node_fwd_cut,
                                                                                        uf=uf)
            if not is_success:
                if graph_changed:
                    self.graph.load_state(graph_state=saved_state)
                failed.add(root_left.id)
                print_state()
                continue

            self.protected_nodes.add(protected_node.id)
            merges = merges_left
            n_merges += len(merges)

            # update on merges
            for i, (a, b) in enumerate(merges_left):
                node_to_cuts.pop(b, default=None)
                if node_to_ok_fwd.pop(b, None):
                    n_ok_fwd -= 1
                if node_to_ok_bwd.pop(b, None):
                    n_ok_bwd -= 1
                if node_to_ok_both.pop(b, None):
                    n_ok_both -= 1

                if b in failed:
                    failed.remove(b)
                    failed_then_merged.add(b)

                uf.union(a, b, smallest_new_root=False)

                if i == len(merges_left) - 1:
                    assert a == protected_node.id
                    node = self.graph[a]

                    was_ok_bwd_b4_merge = node_to_ok_bwd[a]
                    is_ok_bwd_after_merge = not self.cwf.is_comm_bounded_backward(node)
                    node_to_ok_bwd[a] = is_ok_bwd_after_merge
                    node_to_ok_both[a] = is_ok_bwd_after_merge  # its already ok fwd
                    if is_ok_bwd_after_merge and not was_ok_bwd_b4_merge:
                        n_ok_bwd += 1
                    if is_ok_bwd_after_merge:
                        n_ok_both += 1

                    assert not node_to_ok_fwd[a]
                    node_to_ok_fwd[a] = True
                    n_ok_fwd += 1

            # self.update_sorted_edges_on_merges(edges_to_value, merges, allow_poped_outside=True)

            if verbose:
                print_state()

    def update_sorted_edges_on_merges(self, edges_to_value, merges: List[Tuple[int, int]],
                                      allow_poped_outside=True):
        # problem: given (u,v) then v is merged, but we want to remove edges (x,v) or (v,x).
        # solution: union find in the opposite direction to track roots, which are in edge dict.
        # than use self.uf to replace these edges if not removed.

        assert isinstance(edges_to_value, ValueSortedDict)
        uf = self.uf
        uf_bwd = UnionFind()
        for edge in merges:
            # update the overall union find on merges.
            self.uf.union(*edge, smallest_new_root=False)
            uf_bwd.add(edge[0])
            uf_bwd.add(edge[1])
        for (a, b) in merges:
            # find edge to remove
            a_old = uf_bwd[uf_bwd.find(a)]
            b_old = uf_bwd[uf_bwd.find(b)]

            uf_bwd.union(a, b)  # backward

            edge_to_remove = (a_old, b_old)
            try:
                del edges_to_value[edge_to_remove]
            except KeyError as e:
                if not allow_poped_outside:
                    raise e

            # Also update edge weights for b.in_edges

            a_new = uf[uf.find(a)]
            b_new = uf[uf.find(a)]
            cur_b_node = self.graph[b_new]
            cur_a_node = self.graph[a_new]

            for cur_x_node in cur_b_node.out_edges:
                x_new = cur_x_node.id
                x_old = uf_bwd[uf_bwd.find(x_new)] if x_new in uf_bwd else x_new
                edge_to_remove = (b_old, x_old)

                try:
                    del edges_to_value[edge_to_remove]
                except KeyError as e:
                    if not allow_poped_outside:
                        raise e

                # from a as well, we are replacing it, careful, it may not happen.
                edge_to_remove = (a_old, x_old)
                if edge_to_remove in edges_to_value:
                    del edges_to_value[edge_to_remove]
                edge_to_add = (a_new, x_new)
                cur_x_node.update_compound_weights_from_uf(uf)
                value_of_edge_to_add = self.ewf(cur_a_node, cur_x_node)
                # now, do the job
                edges_to_value[edge_to_add] = value_of_edge_to_add

            for cur_x_node in cur_b_node.in_edges:
                x_new = cur_x_node.id
                x_old = uf_bwd[uf_bwd.find(x_new)] if x_new in uf_bwd else x_new
                edge_to_remove = (x_old, b_old)

                try:
                    del edges_to_value[edge_to_remove]
                except KeyError as e:
                    if not allow_poped_outside:
                        raise e

                # from a as well, we are replacing it, careful, it may not happen.
                edge_to_remove = (x_old, a_old)
                if edge_to_remove in edges_to_value:
                    del edges_to_value[edge_to_remove]

                edge_to_add = (x_new, a_new)
                cur_x_node.update_compound_weights_from_uf(uf)
                value_of_edge_to_add = self.ewf(cur_x_node, cur_a_node)
                # now, do the job
                edges_to_value[edge_to_add] = value_of_edge_to_add

    def search_left(self, root_left: Node, comm_objective, uf) -> Tuple[
        bool, bool, List[Tuple[int, int]], Union[Node, None]]:
        # comp: forward + backward
        # comm: forward
        partial_uf = UnionFind()
        merges = []
        graph_changed = False
        assert self.nwf.ratio == 1
        assert self.ewf.ratio == 0

        # early exit, it shouldn't happen though
        current_comp = self.nwf(root_left)
        if current_comp >= comm_objective:
            warnings.warn("early exit without merges shouldn't happen")
            return True, graph_changed, merges, root_left

        cur_merged = root_left
        # backward BFS
        source = root_left
        visited = {source}
        # TODO: order can be topo and not just bfs, e.g edge 4->50 should be taken *after* 49->50
        # even though its same topo order.
        # this requires a "smarted queue" e.g sorted by topo sort distance
        # this also requires topo-sort after each merge, or dynamic toposort.

        queue = deque([(source, reversed(source.args))])
        while queue:
            parent, children = queue[0]
            # if parent not in merged:
            #     continue  # visited, but not merged to source
            try:
                child = next(children)

                if child not in visited:
                    visited.add(child)
                    # TODO: check if we can merge without cycles
                    if child.id in self.protected_nodes or child.type == NodeTypes.IN:
                        continue

                    can_merge_without_cycles = not check_cycle2(self.graph, child, cur_merged)
                    if can_merge_without_cycles:
                        queue.append((child, reversed(child.args)))
                        assert child in cur_merged.args
                        self.graph.merge(child.id, cur_merged.id, edge_weight_function=self.ewf, uf=uf,
                                         partial_uf=partial_uf)
                        self.graph.topo_sort(change_graph=False, resort_edges=False)
                        graph_changed = True

                        partial_uf.add(child.id)
                        partial_uf.add(cur_merged.id)
                        partial_uf.union(child.id, cur_merged.id)

                        merges.append((child.id, cur_merged.id))
                        cur_merged = child
                        current_comp = self.nwf(cur_merged)
                        # update comm objective in case of merged edges
                        comm_objective = sum(self.ewf(cur_merged, nn) for nn in cur_merged.out_edges)

                        if current_comp >= comm_objective:
                            return True, graph_changed, merges, cur_merged
            except StopIteration:
                queue.popleft()

        assert current_comp < comm_objective
        merges = []  # does not really matter, we are using a checkpoint.
        return False, graph_changed, merges, None

    # def search_right(self, root_right: Node, left_root: Node, comm_objective, uf):
    #     partial_uf = UnionFind()
    #     merges = []
    #     protected_changed = False
    #     graph_changed = False
    #     # forward BFS
    #     source = root_right
    #     visited = {source}
    #     assert self.nwf.ratio == 1
    #     current_comp = self.nwf(source)
    #     if current_comp >= comm_objective:
    #         return True, graph_changed, protected_changed, merges
    #     cur_merged = source
    #
    #     queue = deque([(source, iter(source.out_edges))])
    #     while queue:
    #         parent, children = queue[0]
    #         # if parent not in merged:
    #         #     continue  # visited, but not merged to source
    #         try:
    #             child = next(children)
    #
    #             if child not in visited:
    #                 visited.add(child)
    #                 # TODO: check if we can merge without cycles
    #
    #                 edge = (cur_merged.id, child.id)
    #                 if edge in self.protected_edges:
    #                     continue
    #
    #                 can_merge_without_cycles = not check_cycle2(self.graph, cur_merged, child)
    #                 if can_merge_without_cycles:
    #                     queue.append((child, iter(child.out_edges)))
    #                     assert child in cur_merged.out_edges
    #                     self.graph.merge(cur_merged.id, child.id, edge_weight_function=self.ewf, uf=uf,
    #                                      partial_uf=partial_uf)
    #                     merges.append((cur_merged.id, child.id))
    #                     protected_changed |= self.change_protected_before_merge(cur_merged, child)
    #
    #                     partial_uf.add(child.id)
    #                     partial_uf.add(cur_merged.id)
    #                     partial_uf.union(cur_merged.id, child.id)
    #
    #                     graph_changed = True
    #                     # cur_merged = cur_merged
    #                     current_comp = self.nwf(cur_merged)
    #
    #                     # update comm objective in case of merged edges
    #                     tmp = self.ewf.ratio
    #                     self.ewf.ratio = -1
    #                     comm_objective = sum(self.ewf(cur_merged, nn) for nn in cur_merged.out_edges)
    #                     self.ewf.ratio = tmp
    #
    #                     if current_comp >= comm_objective:
    #                         return True, graph_changed, protected_changed, merges
    #         except StopIteration:
    #             queue.popleft()
    #
    #     assert current_comp < comm_objective
    #     return False, graph_changed, protected_changed, merges

    def sorted_graph_forward_edges(self, descending=False) -> Dict[Tuple[Node, Node], float]:

        # TODO: if memory is a problem,  use generator for larger graphs
        edges = list()
        for node in self.graph.non_input_nodes:
            edges.extend([(node, e) for e in node.out_edges])

        if not descending:

            d = ValueSortedDict({
                (e[0].id, e[1].id): self.ewf(*e) for e in edges
            })
        else:

            d = ValueSortedDict(lambda x: -x, {
                (e[0].id, e[1].id): self.ewf(*e) for e in edges
            })

        return d

    def sorted_block_to_cuts(self, forward=True, descending=False) -> Dict[Tuple[Node, Node], float]:
        if forward:
            t = {node.id: sum(self.ewf(node, nn) for nn in node.out_edges) for node in self.graph.non_input_nodes}
        else:
            t = {node.id: self.cwf.calculate_comm_backward([(nn, node) for nn in node.in_edges]) for node in
                 self.graph.non_input_nodes}

        if not descending:
            # TODO: can do a partial relative order with topo-sort ID.
            # taking later blocks first
            d = ValueSortedDict(t)
        else:
            d = ValueSortedDict(lambda x: -x, t)

        return d
