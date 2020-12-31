from collections import deque
from copy import copy
from typing import Set, Dict, Tuple, List

from sortedcollections import ValueSortedDict

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node
from autopipe.autopipe.union_find import UnionFind

# TODO: look at cuts instead of edges
# TODO: take statisic of "handled nodes".

class RatioBlockCreator:
    def __init__(self, graph: Graph, edge_weight_function: EdgeWeightFunction,
                 node_weight_function: NodeWeightFunction, uf: UnionFind):
        self.graph = graph
        self.ewf = edge_weight_function
        self.nwf = node_weight_function
        self.uf = uf

        # edges we already handled and don't want to touch
        # self.merged_nodes = set()  # merge(u,v) <-> v in merged_nodes
        self.protected_edges: Set[int, int] = set()

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
        uf = self.uf
        edges_to_value = self.sorted_graph_forward_edges()
        # TODO: also handle updating the sort, removing merged edges from the graph
        # Needs some union find.
        n_merges = 0
        n_iter = 0

        while edges_to_value and len(self.graph) > L:
            n_iter += 1
            edge, comm_objective_left = edges_to_value.popitem()
            root_left = self.graph[edge[0]]
            root_right = self.graph[edge[1]]

            tmp = self.ewf.ratio
            assert tmp == 0
            self.ewf.ratio = -1
            comm_objective_right = self.ewf(root_left, root_right)
            self.ewf.ratio = tmp

            saved_state = self.graph.state()  # O(N). we are dead...
            protected_copy = copy(self.protected_edges)
            (protect_edge_left_condition, graph_changed, protected_changed, merges1) = self.search_left(root_left,
                                                                                                        root_right,
                                                                                                        comm_objective_left,
                                                                                                        uf=uf)
            if not protect_edge_left_condition:
                if graph_changed:
                    self.graph.load_state(graph_state=saved_state)
                if protected_changed:
                    self.protected_edges = protected_copy
                continue

            (protect_edge_right_condition, graph_changed2, protected_changed2, merges2) = self.search_right(root_right,
                                                                                                            root_left,
                                                                                                            comm_objective_right,
                                                                                                            uf=uf)

            graph_changed = graph_changed or graph_changed2
            if not protect_edge_right_condition:
                if graph_changed:
                    self.graph.load_state(graph_state=saved_state)
                if protected_changed:
                    self.protected_edges = protected_copy
                continue

            self.protected_edges.add(edge)
            merges = merges1 + merges2

            self.update_sorted_edges_on_merges(edges_to_value, merges, allow_poped_outside=True)
            if verbose:
                cur_merges = len(merges)
                # n_merges_left = len(merges1)
                # n_merges_right = len(merges2)
                n_merges += len(merges)
                n_protected = len(self.protected_edges)
                print(
                    f"-I- Iter {n_iter} nodes {self.graph.num_nodes}, merges {n_merges} (now: {cur_merges}), protected: {n_protected}, remaining {len(edges_to_value)}")

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

    def search_left(self, root_left: Node, right_root: Node, comm_objective, uf):
        # comp: forward + backward
        # comm: forward
        partial_uf = UnionFind()
        merges = []
        graph_changed = False
        protected_changed = False
        # backward BFS
        source = root_left
        visited = {source}
        assert self.nwf.ratio == 1
        current_comp = self.nwf(source)
        if current_comp >= comm_objective:
            return True, graph_changed, protected_changed, merges
        cur_merged = root_left

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
                    edge = (child.id, cur_merged.id)
                    if edge in self.protected_edges:
                        continue

                    can_merge_without_cycles = not check_cycle2(self.graph, child, cur_merged)
                    if can_merge_without_cycles:
                        queue.append((child, reversed(child.args)))
                        assert child in cur_merged.args
                        self.graph.merge(child.id, cur_merged.id, edge_weight_function=self.ewf, uf=uf,
                                         partial_uf=partial_uf)
                        graph_changed = True

                        partial_uf.add(child.id)
                        partial_uf.add(cur_merged.id)
                        partial_uf.union(child.id, cur_merged.id)

                        protected_changed |= self.change_protected_before_merge(child, cur_merged)
                        merges.append((child.id, cur_merged.id))
                        cur_merged = child
                        current_comp = self.nwf(cur_merged)
                        # update comm objective in case of merged edges
                        comm_objective = self.ewf(cur_merged, right_root)

                        if current_comp >= comm_objective:
                            return True, graph_changed, protected_changed, merges
            except StopIteration:
                queue.popleft()

        assert current_comp < comm_objective
        return False, graph_changed, protected_changed, merges

    def search_right(self, root_right: Node, left_root: Node, comm_objective, uf):
        partial_uf = UnionFind()
        merges = []
        protected_changed = False
        graph_changed = False
        # forward BFS
        source = root_right
        visited = {source}
        assert self.nwf.ratio == 1
        current_comp = self.nwf(source)
        if current_comp >= comm_objective:
            return True, graph_changed, protected_changed, merges
        cur_merged = source

        queue = deque([(source, iter(source.out_edges))])
        while queue:
            parent, children = queue[0]
            # if parent not in merged:
            #     continue  # visited, but not merged to source
            try:
                child = next(children)

                if child not in visited:
                    visited.add(child)
                    # TODO: check if we can merge without cycles

                    edge = (cur_merged.id, child.id)
                    if edge in self.protected_edges:
                        continue

                    can_merge_without_cycles = not check_cycle2(self.graph, cur_merged, child)
                    if can_merge_without_cycles:
                        queue.append((child, iter(child.out_edges)))
                        assert child in cur_merged.out_edges
                        self.graph.merge(cur_merged.id, child.id, edge_weight_function=self.ewf, uf=uf,
                                         partial_uf=partial_uf)
                        merges.append((cur_merged.id, child.id))
                        protected_changed |= self.change_protected_before_merge(cur_merged, child)

                        partial_uf.add(child.id)
                        partial_uf.add(cur_merged.id)
                        partial_uf.union(cur_merged.id, child.id)

                        graph_changed = True
                        # cur_merged = cur_merged
                        current_comp = self.nwf(cur_merged)

                        # update comm objective in case of merged edges
                        tmp = self.ewf.ratio
                        self.ewf.ratio = -1
                        comm_objective = self.ewf(left_root, cur_merged)
                        self.ewf.ratio = tmp

                        if current_comp >= comm_objective:
                            return True, graph_changed, protected_changed, merges
            except StopIteration:
                queue.popleft()

        assert current_comp < comm_objective
        return False, graph_changed, protected_changed, merges

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
