from pytorch_Gpipe.model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction
from .data_structures import QuotientGraph, SimpleNode, PriorityQueue, VerticeStageConnections
from .gpa import coarsening, refine
import random
import math
import numpy as np
from typing import Tuple, Dict, Optional
from collections import defaultdict, namedtuple
import enum
from multiprocessing import Pool
import time
import os

###################################################################################################


class META_ALGORITH(enum.Enum):
    SINGLE_LEVEL = 1
    MULTI_LEVEL = 2

    def __repr__(self):
        return self.name


class ALGORITHM(enum.Enum):
    SIMPLE_MOVES = 1
    ADVANCED_MOVES = 2
    GLOBAL_MOVES = 3
    FIDUCCIA_MATTHEYSES_MOVES = 4

    def __repr__(self):
        return self.name


class Objective(enum.Enum):
    EDGE_CUT = 1
    STAGE_TIME = 2

    def __repr__(self):
        return self.name


###################################################################################################

#create initial partitioning by taking consecutive blocks of equal weight
#the blocks are consecutive blocks of nodes acquired by a Khan's algorithm
def initial_divide(graph: Graph, k: int,
                   node_weights: Dict[SimpleNode, float])->Tuple[int,...]:
    random_topo_sort = random_Khan_algorithm(graph)
    node_weights = np.asarray([node_weights[n] for n in random_topo_sort])
    cumulative_node_weights = np.cumsum(node_weights)

    total_weight = cumulative_node_weights[-1]
    avg_weight = total_weight / k

    Vs = []

    options = [math.floor(avg_weight), math.ceil(avg_weight)]
    acc = 0
    # k partitions require k-1 seperators
    while len(Vs) < k - 1:
        stage_weight = options[random.randint(0, 1)]
        acc += stage_weight
        Vs.append(np.searchsorted(cumulative_node_weights, acc))

    idxs = [-1] + Vs + [len(cumulative_node_weights) - 1]

    idxs = list(zip(map(lambda i: i + 1, idxs), idxs[1:]))

    order = [n.id for n in random_topo_sort]

    #set partitioning
    for i, (start, end) in enumerate(idxs):
        for n in random_topo_sort[start:end + 1]:
            n.stage_id = i

    return tuple(order)


# create a random topological sorting
def random_Khan_algorithm(graph: Graph):
    S = []
    T = []

    degs = dict()
    nodes = list(graph.nodes)
    random.shuffle(nodes)

    # o(V)
    for n in nodes:
        if len(n.in_edges) == 0:
            S.append(n)
        else:
            degs[n] = len(n.in_edges)
    # O(E)
    while S:
        idx = random.randint(0, len(S) - 1)
        n = S[idx]
        del S[idx]
        T.append(n)
        for o in n.out_edges:
            degs[o] -= 1
            if degs[o] == 0:
                S.append(o)

    #if we have a cycle S will be empty and T will not contain all of the nodes
    assert len(T) == len(nodes),"cycle detected"
    return T


###################################################################################################


# move nodes between adjacent partitions as long as edge cut improves and constraints are enforced
# aka i=>i+1 or i=>i+1
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def simple_moves(partition_volumes: Dict[int, float],
                 edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                 node_weights: Dict[SimpleNode, float],
                 L_max: float,
                 rounds: int = 1):
    connections = VerticeStageConnections(node_weights)

    def update_function(v, dst):
        partition_volumes[v.stage_id] -= node_weights[v]
        partition_volumes[dst] += node_weights[v]
        connections.move_node(v, dst)


    state = PartitionState(edge_weights, node_weights, partition_volumes,
                           L_max)

    # we use 0 based indexing
    k = len(partition_volumes) - 1

    nodes = list(node_weights.keys())
    for _ in range(rounds):
        changed = False

        random.shuffle(nodes)
        # O(E)
        for n in nodes:
            gain_left = -np.inf
            if (n.stage_id > 0
                ) and (not connections.has_in_connection(n, n.stage_id)) and (
                    (partition_volumes[n.stage_id - 1] + node_weights[n]) < L_max):
                gain_left = calculate_edge_gain(n, n.stage_id - 1, state)

            gain_right = -np.inf
            if (n.stage_id < k
                ) and (not connections.has_out_connection(n, n.stage_id)) and (
                    (partition_volumes[n.stage_id + 1] + node_weights[n]) < L_max):
                gain_right = calculate_edge_gain(n, n.stage_id + 1, state)

            moves = defaultdict(list)
            moves[gain_left].append(n.stage_id - 1)
            moves[gain_right].append(n.stage_id + 1)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue

            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves, 1)[0]

            update_function(n, dst)
            n.stage_id = dst

        if not changed:
            break


# move nodes between all partitions as long as edge cut improves and constraints are enforced
# uses a sufficient condition for enforcing acyclicicity not a necessary condition
# as such some options are skipped
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def advanced_moves(partition_volumes: Dict[int, float],
                   edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                   node_weights: Dict[SimpleNode, float],
                   L_max: float,
                   rounds: int = 1):

    def update_function(v, dst):
        partition_volumes[v.stage_id] -= node_weights[v]
        partition_volumes[dst] += node_weights[v]


    state = PartitionState(edge_weights, node_weights, partition_volumes,
                           L_max)

    nodes = list(node_weights.keys())
    for _ in range(rounds):
        changed = False

        random.shuffle(nodes)
        # O(E)
        for n in nodes:
            # [A,B] is the eligible partition range that n can be placed in
            A = max((i.stage_id for i in n.in_edges), default=n.stage_id)
            B = min((o.stage_id for o in n.out_edges), default=n.stage_id)

            if A == B:
                # n has in and out connections in it's partition
                # cannot be moved
                continue

            moves = defaultdict(list)
            for dst in range(A, B + 1):
                if dst == n.stage_id:
                    continue
                edge_gain = calculate_edge_gain(n, dst, state)
                if (partition_volumes[dst] + node_weights[n]) > L_max:
                    edge_gain = -np.inf
                moves[edge_gain].append(dst)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue

            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves, 1)[0]

            update_function(n, dst)
            n.stage_id = dst

        if not changed:
            break


# move nodes between all partitions as long as edge cut improves and constraints are enforced
# uses Khan's algorithm to ensure we do not create cycles in the quotient graph
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def global_moves(partition_volumes: Dict[int, float],
                 edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                 node_weights: Dict[SimpleNode, float],
                 L_max: float,
                 rounds: int = 1):

    def update_function(v, dst):
        partition_volumes[v.stage_id] -= node_weights[v]
        partition_volumes[dst] += node_weights[v]

    state = PartitionState(edge_weights, node_weights, partition_volumes,
                           L_max)

    quotient_graph = QuotientGraph(node_weights.keys())
    nodes = list(node_weights.keys())

    for _ in range(rounds):
        changed = False
        random.shuffle(nodes)
        # O(E(k+mq))
        for n in nodes:
            moves = defaultdict(list)
            for dst in partition_volumes.keys():
                if dst == n.stage_id:
                    continue

                gain = calculate_edge_gain(n, dst, state)
                if ((partition_volumes[dst] + node_weights[n]) >
                        L_max) or quotient_graph.move_creates_cycle(n, dst):
                    gain = -np.inf

                moves[gain].append(dst)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue

            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves, 1)[0]

            update_function(n, dst)
            quotient_graph.move_node(n, dst)

        if not changed:
            break


# move nodes between partitions
# moves with negative gain are also eligible in order to escape local minima
# the partitioning with the best objective will be returned
def Fiduccia_Mattheyses_moves(partition_volumes: Dict[int, float],
                              edge_weights: Dict[Tuple[SimpleNode, SimpleNode],
                                                 float],
                              node_weights: Dict[SimpleNode, float],
                              L_max: float,
                              rounds: int = 1):

    # track which nodes belong to each partiiton
    partitions = {i: set() for i in partition_volumes}
    for n in node_weights:
        partitions[n.stage_id].add(n)

    def update_function(v, dst):
        partition_volumes[v.stage_id] -= node_weights[v]
        partition_volumes[dst] += node_weights[v]
        partitions[v.stage_id].discard(v)
        partitions[dst].add(v)


    state = PartitionState(edge_weights, node_weights, partition_volumes,
                           L_max)

    best_objective = calculate_edge_cut(edge_weights)

    all_blocks = list(partition_volumes.keys())

    for _ in range(rounds):
        active_blocks = set(all_blocks)

        # outer pass
        while active_blocks:
            # select A and B such that A < B and at least one of A,B is active
            A, B = random.sample(all_blocks, 2)
            if A > B:
                A, B = B, A
            if A == B or not (A in active_blocks or B in active_blocks):
                continue
            active_blocks.discard(A)
            active_blocks.discard(B)

            candidate_moves = PriorityQueue()
            for node in partitions[A]:
                if all(o.stage_id >= B for o in node.out_edges):
                    candidate_moves.push_task(calculate_edge_gain(node, B, state),
                                              (node, B))
            for node in partitions[B]:
                if all(i.stage_id <= A for i in node.in_edges):
                    candidate_moves.push_task(calculate_edge_gain(node, A, state),
                                              (node, A))

            locked_nodes = set()
            moves_to_best = dict()
            current_objective = best_objective

            # inner pass
            while len(candidate_moves) > 0:
                node, dst = candidate_moves.pop_task()
                # check if the move is still valid
                if node in locked_nodes:
                    continue
                elif (partition_volumes[dst] + node_weights[node]) > L_max:
                    continue
                elif (node.stage_id == A) and any(o.stage_id < B
                                              for o in node.out_edges):
                    continue
                elif (node.stage_id == B) and any(i.stage_id > A
                                              for i in node.in_edges):
                    continue

                locked_nodes.add(node)

                gain = calculate_edge_gain(node, dst, state)
                current_objective -= gain
                src = node.stage_id

                update_function(node, dst)
                node.stage_id = dst

                if current_objective < best_objective:
                    best_objective = current_objective
                    moves_to_best.clear()
                    active_blocks.add(A)
                    active_blocks.add(B)
                else:
                    # if we did not improve the objective, record the move so we can revert it
                    # at the end of the inner pass
                    moves_to_best[node] = src

                # check if we enabled more moves
                if src == A:
                    for i in node.in_edges:
                        if i.stage_id == A and all(o.stage_id >= B
                                               for o in i.out_edges):
                            gain = calculate_edge_gain(i, B, state)
                            candidate_moves.push_task(gain, (i, B))
                else:
                    for o in node.out_edges:
                        if o.stage_id == B and all(i.stage_id <= A
                                               for i in o.in_edges):
                            gain = calculate_edge_gain(o, A, state)
                            candidate_moves.push_task(gain, (o, A))

            # end of inner pass revert partition to best partition
            for n, dst in moves_to_best.items():
                update_function(n, dst)
                n.stage_id = dst


###################################################################################################


PartitionState = namedtuple(
    "PartitionState", "edge_weights node_weights partition_volumes L_max")


###################################################################################################


def calculate_edge_gain(v: SimpleNode, dest: int,
                        state: PartitionState) -> float:
    edge_weights = state.edge_weights
    gain = 0
    stages = set()
    for n in v.in_edges:
        if n.stage_id in stages:
            continue
        stages.add(n.stage_id)
        if n.stage_id == dest:
            gain += edge_weights[(n, v)]
        elif n.stage_id == v.stage_id:
            gain -= edge_weights[(n, v)]

    stages.clear()
    for n in v.out_edges:
        if n.stage_id in stages:
            continue
        stages.add(n.stage_id)
        if n.stage_id == dest:
            gain += edge_weights[(v, n)]
        elif n.stage_id == v.stage_id:
            gain -= edge_weights[(v, n)]

    return gain


###################################################################################################


def calculate_edge_cut(edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float]) -> float:
    edge_cut = 0

    visited = set()
    for (u,v),w in edge_weights.items():
        if (u.id,v.stage_id) not in visited:
            edge_cut += w
            visited.add((u.id,v.stage_id))

    return edge_cut


def calculate_partition_volumes(
        k: int, node_weights: Dict[SimpleNode, float]) -> Dict[int, float]:
    partition_volumes = {i: 0 for i in range(k)}
    for n, w in node_weights.items():
        partition_volumes[n.stage_id] += w

    return partition_volumes


def calculate_stage_times(
    node_weights: Dict[SimpleNode, float],
    edge_weights: Dict[Tuple[SimpleNode, SimpleNode],
                       float]) -> Dict[int, float]:
    stage_times = defaultdict(lambda: 0)

    for n, w in node_weights.items():
        stage_times[n.stage_id] += w

        # record sent activation only once per destination stage
        destinations = set()
        for o in n.out_edges:
            if (o.stage_id == n.stage_id) or (o.stage_id in destinations):
                continue
            e = edge_weights[(n, o)]
            destinations.add(o.stage_id)
            stage_times[n.stage_id] += e
            stage_times[o.stage_id] += e

    return dict(stage_times)


###################################################################################################


HEURISTICS = {
    ALGORITHM.SIMPLE_MOVES: simple_moves,
    ALGORITHM.ADVANCED_MOVES: advanced_moves,
    ALGORITHM.GLOBAL_MOVES: global_moves,
    ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES: Fiduccia_Mattheyses_moves
}


###################################################################################################


def acyclic_partition(
        graph: Graph,
        k: int,
        epsilon: float = 0.1,
        node_weight_function: Optional[NodeWeightFunction] = None,
        edge_weight_function: Optional[EdgeWeightFunction] = None,
        meta_algorithm: META_ALGORITH = META_ALGORITH.SINGLE_LEVEL,
        objective: Objective = Objective.EDGE_CUT,
        rounds: int = 10,
        allocated_seconds: int = 20,
        use_layers_graph: bool = True
) -> Graph:

    if node_weight_function is None:
        node_weight_function = DefaultWeightFunction()
    if edge_weight_function is None:
        edge_weight_function = DefaultEdgeWeightFunction()


    if use_layers_graph:
        work_graph,lookup = graph.layers_graph()
    else:
        work_graph = graph

    worker_args = [
        dict(graph=work_graph._remove_parallel_edges().state(),
             k=k,
             meta_algorithm=meta_algorithm,
             algorithm=alg,
             epsilon=epsilon,
             node_weight_function=node_weight_function,
             edge_weight_function=edge_weight_function,
             rounds=rounds,
             allocated_seconds=allocated_seconds,
             objective=objective) for alg in ALGORITHM
    ]

    with Pool(len(worker_args)) as pool:
        results = pool.map(worker, worker_args)

    assert len(results) == len(worker_args)

    best_solution,edge_cut,worst_case = None, np.inf, np.inf

    for s,e,w in results:
        if is_better_solution((e,w),(edge_cut,worst_case),objective):
            best_solution = s
            edge_cut = e
            worst_case = w

    for n in work_graph.nodes:
        n.stage_id = best_solution[n.id]

    if use_layers_graph:
        graph.induce_layer_partition(work_graph,lookup)

    assert graph.n_stages == k

    return graph


def worker(kwargs) -> Tuple[Dict[int,int],float,float]:
    graph = Graph(None, None, 
                        None, None, None).load_state(kwargs.pop('graph'))
    kwargs['graph'] = graph
    meta_algorithm = kwargs.pop("meta_algorithm")
    allocated_seconds = kwargs.pop("allocated_seconds")
    objective = kwargs['objective']

    best_solution,edge_cut,worst_case = None, np.inf, np.inf

    nwf = kwargs.pop("node_weight_function")
    ewf = kwargs.pop("edge_weight_function")
    node_weights = dict()
    edge_weights = dict()
    for u in graph.nodes:
        node_weights[u] = nwf(u)
        for o in u.out_edges:
            edge_weights[(u,o)] = ewf(u,o)

    kwargs['node_weights'] = node_weights
    kwargs['edge_weights'] = edge_weights

    start = time.time()
    steps = 0

    while (time.time() - start) < allocated_seconds:
        seed = int.from_bytes(os.urandom(4), byteorder='little')
        random.seed(seed)
        
        if meta_algorithm is META_ALGORITH.SINGLE_LEVEL:
            solution,solution_edge_cut,solution_worst_case = single_level_partitioning(**kwargs)
        else:
            solution,solution_edge_cut,solution_worst_case = multilevel_partitioning(**kwargs)

        if is_better_solution((solution_edge_cut,solution_worst_case),(edge_cut,worst_case),objective):
            best_solution = solution
            edge_cut = solution_edge_cut
            worst_case = solution_worst_case
        steps += 1
    

    return best_solution,edge_cut,worst_case


def is_better_solution(solution:Tuple[float,float],best_solution:Tuple[float,float],objective:Objective)->bool:
    solution_edge_cut,solution_worst_case = solution
    best_edge_cut,best_worst_case = best_solution

    better_edge_cut = solution_edge_cut < best_edge_cut
    better_worst_case = solution_worst_case < best_worst_case

    if objective is Objective.EDGE_CUT:
        return better_edge_cut or ((solution_edge_cut == best_edge_cut) and better_worst_case)

    return better_worst_case or ((solution_worst_case == best_worst_case) and better_edge_cut)


###################################################################################################


def single_level_partitioning(
    graph: Graph,
    node_weights: Dict[SimpleNode,float],
    edge_weights: Dict[Tuple[SimpleNode,SimpleNode],float],
    algorithm: ALGORITHM = ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES,
    k: int = 4,
    epsilon: float = 0.1,
    objective: Objective = Objective.EDGE_CUT,
    rounds: int = 10
) -> Tuple[Dict[int,int],float,float]:
    
    initial_divide(graph, k, node_weights)

    partition_volumes = calculate_partition_volumes(k, node_weights)
    L_max = (1 + epsilon) * math.ceil(sum(partition_volumes.values()) / k)

    
    msg = "\n".join([
        "-I- balanced partitioning is not possible",
        f"   max allowed weight: {L_max:.2f}",
        f"   max node weight: {max(node_weights.values()):.2f}"
    ])

    assert all((v <= L_max for v in node_weights.values())), msg

    # NOTE we optimize the more stable edge cut objective
    # optimizing stage times directly is unstable and can create less partitions than requested
    # when selecting the best solution it could be according to the stage time objective 
    HEURISTICS[algorithm](partition_volumes,
                        edge_weights,
                        node_weights,
                        L_max,
                        rounds=rounds)

    #refine partition in a greedy fashion
    global_moves(partition_volumes,
                edge_weights,
                node_weights,
                L_max,
                rounds=1)

    if objective is Objective.STAGE_TIME:
        partition_volumes = calculate_stage_times(node_weights,edge_weights)

    edge_cut = calculate_edge_cut(edge_weights)

    return {n.id:n.stage_id for n in graph.nodes},edge_cut,max(partition_volumes.values())


def multilevel_partitioning(
                        graph: Graph,
                        node_weights: Dict[SimpleNode,float],
                        edge_weights: Dict[Tuple[SimpleNode,SimpleNode],float],
                        algorithm: ALGORITHM = ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES,
                        k: int = 4,
                        epsilon: float = 0.1,
                        objective: Objective = Objective.EDGE_CUT,
                        rounds: int = 10
                        ) -> Tuple[Dict[int,int],float,float]:
    # NOTE we optimize the more stable edge cut objective
    # optimizing stage times directly is unstable and can create less partitions than requested
    # when selecting the best solution it could be according to the stage time objective 
    single_level_partitioning(graph,
                                node_weights=node_weights,
                                edge_weights=edge_weights,
                                algorithm=algorithm,
                                k=k,
                                epsilon=epsilon,
                                objective=Objective.EDGE_CUT,
                                rounds=rounds)
    
    partition_volumes = calculate_partition_volumes(k, node_weights)
    L_max = (1 + epsilon) * math.ceil(sum(partition_volumes.values()) / k)

    hierarchy = coarsening(graph, node_weights, edge_weights)
    # iterate in reverse order to coarsening
    # from smallest graph to largest graph
    for fine_graph, matching, coarse_graph in reversed(hierarchy):
        HEURISTICS[algorithm](partition_volumes,
                              coarse_graph._edge_weights,
                              coarse_graph._node_weights,
                              L_max,
                              rounds=rounds)
        refine(fine_graph, coarse_graph, matching)

    #update original graph
    root = hierarchy[0][0]
    for i in range(len(graph)):
        graph[i].stage_id = root[i].stage_id
    
    if objective is Objective.STAGE_TIME:
        partition_volumes = calculate_stage_times(node_weights,edge_weights)

    edge_cut = calculate_edge_cut(edge_weights)

    return {n.id:n.stage_id for n in graph.nodes},edge_cut,max(partition_volumes.values())


###################################################################################################


class DefaultWeightFunction():
    def __call__(self,u:SimpleNode)->float:
        return 1

class DefaultEdgeWeightFunction():
    def __call__(self,u:SimpleNode,v:SimpleNode)->float:
        return 1