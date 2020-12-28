import warnings
from collections import defaultdict
from typing import Iterable

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction, \
    CoarsenedWeightFunction
from autopipe.autopipe.model_partitioning.utils import re_assign_partition_indices
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node


class Refiner:
    def __init__(self, graph: Graph, node_weight_function: NodeWeightFunction,
                 edge_weight_function: EdgeWeightFunction):
        u = graph.unique_partitions_ids
        if None in u:
            raise NotImplementedError("please remove None stage_id")
        n_stages = len(u)
        assert min(u) == 0
        assert max(u) == n_stages - 1

        cwf = CoarsenedWeightFunction(edge_weight_function=edge_weight_function,
                                      node_weight_function=node_weight_function)

        self.nwf = node_weight_function
        self.ewf = edge_weight_function
        self.cwf = cwf
        self.graph = graph
        self.n_stages = n_stages

        stage_to_nodes = defaultdict(set)
        for n in graph.non_input_nodes:
            stage_to_nodes[n.stage_id].add(n)
        self.stage_to_nodes = stage_to_nodes

        stage_to_gpu = {stage_id: next(iter(nodes)).gpu_id for stage_id, nodes in stage_to_nodes.items()}
        self.stage_to_gpu = stage_to_gpu

        u = set(stage_to_gpu.values())
        assert len(u) - 1 == max(u)
        assert min(u) == 0
        num_gpus = len(u)
        self.num_gpus = num_gpus

        # Note: summing fwd+bwd, ratio=1 TODO: ratio!=1
        stage_to_split_comp_cost = {stage_id: cwf.calculate_comp(nodes) for stage_id, nodes in stage_to_nodes.items()}
        stage_to_comp_cost = {stage_id: sum(v) for stage_id, v in stage_to_split_comp_cost.items()}
        self.stage_to_comp_cost = stage_to_comp_cost
        self.stage_to_split_comp_cost = stage_to_split_comp_cost

        gpu_to_comp_cost = {gpu_id: 0 for gpu_id in range(num_gpus)}
        for stage_id, nodes in stage_to_nodes.items():
            gpu_id = stage_to_gpu[stage_id]
            gpu_to_comp_cost[gpu_id] += stage_to_comp_cost[stage_id]
        self.gpu_to_comp_cost = gpu_to_comp_cost

        stage_borders = dict()
        for stage_id, nodes in stage_to_nodes.items():
            stage_borders[stage_id] = self.cwf.calculate_borders(nodes)
        self.stage_borders = stage_borders

        self.stage_to_cost = self.calc_stage_to_cost()
        self.best_objective = self.calc_objective()

    def calc_stage_to_cost(self):
        cwf = self.cwf
        gpu_to_comp_cost = self.gpu_to_comp_cost
        stage_borders = self.stage_borders
        stage_to_nodes = self.stage_to_nodes
        stage_to_gpu = self.stage_to_gpu

        stage_to_cost = {
            stage_id: cwf(nodes, boarders=stage_borders[stage_id],
                          total_gpu_comp_cost=gpu_to_comp_cost[stage_to_gpu[stage_id]],
                          total_stage_comp_cost_fwd=self.stage_to_split_comp_cost[stage_id][0],
                          total_stage_comp_cost_bwd=self.stage_to_split_comp_cost[stage_id][1]
                          ) for
            stage_id, nodes in stage_to_nodes.items()}
        return stage_to_cost

    def calc_objective(self):
        # to minimize
        return max(self.stage_to_cost.values())

    def update_on_move(self, nodes: Iterable[Node], new_stage_id: int, escape_minima=False):
        prev_stage_id = next(iter(nodes)).stage_id

        # Move
        self._apply_move(nodes, new_stage_id)

        # Check objective: is better solution?
        new_objective = self.calc_objective()
        if new_objective < self.best_objective:
            self.best_objective = new_objective
            return True
        elif new_objective == self.best_objective:
            self.best_objective = new_objective
            # TODO: check if communication decreased
            comm_sign = 0  # prev - now
            if comm_sign > 0 or (comm_sign == 0 and escape_minima):
                return True

        # Undo
        self._apply_move(nodes, prev_stage_id)
        return False

    def _apply_move(self, nodes: Iterable[Node], new_stage_id: int):
        prev_stage_id = next(iter(nodes)).stage_id
        # assert new_stage_id >= prev_stage_id  # TODO: concurrent stages
        prev_gpu_id = self.stage_to_gpu[prev_stage_id]  # optionally: assert
        new_gpu_id = self.stage_to_gpu[new_stage_id]
        nwf = self.nwf
        for node in nodes:
            node.stage_id = new_stage_id
            node.gpu_id = new_gpu_id
            self.stage_to_nodes[prev_stage_id].remove(node)
            self.stage_to_nodes[new_stage_id].add(node)
            if len(self.stage_to_nodes[prev_stage_id]) == 0:
                warnings.warn(f"stage {prev_stage_id} eliminated in refinement")

            tmp = nwf.ratio
            assert tmp == 1

            nwf.ratio = 0
            comp_cost_fwd = nwf(node)
            nwf.ratio = -1
            comp_cost_bwd = nwf(node)

            nwf.ratio = 1

            comp_cost = comp_cost_bwd + comp_cost_fwd

            # comp_cost = self.nwf(node)
            self.stage_to_comp_cost[prev_stage_id] -= comp_cost
            self.stage_to_comp_cost[new_stage_id] += comp_cost

            x = self.stage_to_split_comp_cost[prev_stage_id]
            self.stage_to_split_comp_cost[prev_stage_id] = (x[0] - comp_cost_fwd, x[1] - comp_cost_bwd)

            y = self.stage_to_split_comp_cost[new_stage_id]
            self.stage_to_split_comp_cost[new_stage_id] = (y[0] + comp_cost_fwd, y[1] + comp_cost_bwd)

            self.gpu_to_comp_cost[prev_gpu_id] -= comp_cost
            self.gpu_to_comp_cost[new_gpu_id] += comp_cost
        # Stage borders
        self.stage_borders[prev_stage_id] = self.cwf.calculate_borders(self.stage_to_nodes[prev_stage_id])
        self.stage_borders[new_stage_id] = self.cwf.calculate_borders(self.stage_to_nodes[new_stage_id])
        # TODO: more efficient implementation for new stage_borders, but it can get tricky.
        # outgoing_edges, outgoing_nodes, incoming_edges, incoming_nodes = self.stage_borders
        self.stage_to_cost = self.calc_stage_to_cost()

    @staticmethod
    def is_move_valid_local(node):
        # initial validity:
        cur_stage = node.stage_id
        others = [nn.stage_id for nn in node.out_edges]
        for i in others:
            if i == cur_stage:
                return False
        return True

    @staticmethod
    def is_move_valid_topo(node, stage_id):
        others = [nn.stage_id for nn in node.out_edges]
        for i in others:
            if i > stage_id:  # TODO: concurent stages?
                return False
        return True


def refine(graph: Graph, node_weight_function: NodeWeightFunction, edge_weight_function: EdgeWeightFunction,
           round_limit=-1):
    # O(L*N) = O(P*N)
    # reassing partition indices to topo order
    re_assign_partition_indices(graph)
    refiner = Refiner(graph, node_weight_function, edge_weight_function)
    #
    # graph_to_objective = dict()
    # graph_to_objective[graph] = refiner.calc_objective()
    #

    # refinement moves
    rounds = 0
    num_moved = 1
    total_moves = 0
    while num_moved > 0 and (round_limit < 0 or round_limit < rounds):
        rounds += 1
        num_moved = 0
        for stage_id, borders in reversed(refiner.stage_borders.items()):
            outgoing_edges, outgoing_nodes, incoming_edges, incoming_nodes = borders

            invalid_local_nodes = set()
            valid_local_noedes = set()
            moved_local = set()

            # Outgoing:
            for e in sorted(outgoing_edges, key=lambda x: (x[0].id, x[1].id), reverse=True):
                node = e[0]
                dst_stage = e[1].stage_id

                if node not in valid_local_noedes:
                    if node not in valid_local_noedes and (node in invalid_local_nodes):
                        if refiner.is_move_valid_local(node):
                            valid_local_noedes.add(node)
                        else:
                            invalid_local_nodes.add(node)
                            continue

                if not refiner.is_move_valid_topo(node, dst_stage):
                    continue

                moved = refiner.update_on_move(nodes=[node], new_stage_id=dst_stage, escape_minima=False)
                if moved:
                    moved_local.add(node)
                    num_moved += 1
        total_moves += num_moved
        print(f"Round {rounds}: num_moved {num_moved}")

    print(f"Refinement ended after {rounds} rounds and {total_moves} moves")
    # try invalids? next round
    # ingoing? taken care by outgoin!
    # TODO: try with merges?
