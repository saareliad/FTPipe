import warnings
from collections import defaultdict
from typing import Iterable

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction, \
    CoarsenedWeightFunction
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

        stage_to_gpu = {stage_id: next(iter(nodes))[0].gpu_id for stage_id, nodes in stage_to_nodes.items()}
        self.stage_to_gpu = stage_to_gpu

        u = set(stage_to_gpu.values())
        assert len(u) - 1 == max(u)
        assert min(u) == 0
        num_gpus = len(u)
        self.num_gpus = num_gpus

        stage_to_comp_cost = {stage_id: cwf.calculate_comp(nodes) for stage_id, nodes in stage_to_nodes.items()}
        self.stage_to_comp_cost = stage_to_comp_cost

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

        stage_to_cost = {
            stage_id: cwf(nodes, boarders=stage_borders[stage_id], total_gpu_comp_cost=gpu_to_comp_cost[stage_id]) for
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
            return
        elif new_objective == self.best_objective:
            self.best_objective = new_objective
            # TODO: check if communication decreased
            comm_sign = 0  # prev - now
            if comm_sign > 0 or (comm_sign == 0 and escape_minima):
                return

        # Undo
        self._apply_move(nodes, prev_stage_id)

    def _apply_move(self, nodes: Iterable[Node], new_stage_id: int):
        prev_stage_id = next(iter(nodes)).stage_id
        # assert new_stage_id >= prev_stage_id  # TODO: concurrent stages
        prev_gpu_id = self.stage_to_gpu[prev_stage_id]  # optionally: assert
        new_gpu_id = self.stage_to_gpu[new_stage_id]
        for node in nodes:
            node.stage_id = new_stage_id
            node.gpu_id = new_gpu_id
            self.stage_to_nodes[prev_stage_id].remove(node)
            self.stage_to_nodes[new_stage_id].add(node)
            if len(self.stage_to_nodes[prev_stage_id]) == 0:
                warnings.warn(f"stage {prev_stage_id} eliminated in refinement")

            comp_cost = self.nwf(node)
            self.stage_to_comp_cost[prev_stage_id] -= comp_cost
            self.stage_to_comp_cost[new_stage_id] += comp_cost

            self.gpu_to_comp_cost[prev_gpu_id] -= comp_cost
            self.gpu_to_comp_cost[new_stage_id] += comp_cost
        # Stage borders
        self.stage_borders[prev_gpu_id] = self.cwf.calculate_borders(self.stage_to_nodes[prev_stage_id])
        self.stage_borders[new_gpu_id] = self.cwf.calculate_borders(self.stage_to_nodes[new_stage_id])
        # TODO: more efficient implementation for new stage_borders, but it can get tricky.
        # outgoing_edges, outgoing_nodes, incoming_edges, incoming_nodes = self.stage_borders
        self.stage_to_cost = self.calc_stage_to_cost()


def refine(graph: Graph, node_weight_function: NodeWeightFunction, edge_weight_function: EdgeWeightFunction):
    # TODO: reassing partition indices to topo order
    refiner = Refiner(graph, node_weight_function, edge_weight_function)
    #
    # graph_to_objective = dict()
    # graph_to_objective[graph] = refiner.calc_objective()
    #

    # TODO: refinement moves
