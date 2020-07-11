from pytorch_Gpipe.model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction
from .data_structures import QuotientGraph, SimpleNode, PriorityQueue, DoublePriority, VerticeStageConnections
from .gpa import coarsening, refine
import random
import math
import numpy as np
from typing import Tuple, Dict, Optional, Iterator,List
from collections import defaultdict, namedtuple
from itertools import chain,islice
import enum
from multiprocessing import Pool
import time
import os
import json

DEBUG = False
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
                 rounds: int = 1,
                 objective: Objective = Objective.EDGE_CUT):
    connections = VerticeStageConnections(node_weights)

    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain

        def update_function(v, dst):
            partition_volumes[v.stage_id] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
            connections.move_node(v, dst)
    else:
        gain_function = calculate_stage_time_gain

        def update_function(v, dst):
            update_stage_times(v, dst, node_weights, edge_weights,
                               partition_volumes)
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
                gain_left = gain_function(n, n.stage_id - 1, state)

            gain_right = -np.inf
            if (n.stage_id < k
                ) and (not connections.has_out_connection(n, n.stage_id)) and (
                    (partition_volumes[n.stage_id + 1] + node_weights[n]) < L_max):
                gain_right = gain_function(n, n.stage_id + 1, state)

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
                   rounds: int = 1,
                   objective: Objective = Objective.EDGE_CUT):

    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain

        def update_function(v, dst):
            partition_volumes[v.stage_id] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
    else:
        gain_function = calculate_stage_time_gain

        def update_function(v, dst):
            update_stage_times(v, dst, node_weights, edge_weights,
                               partition_volumes)

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
                edge_gain = gain_function(n, dst, state)
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
                 rounds: int = 1,
                 objective: Objective = Objective.EDGE_CUT):

    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain

        def update_function(v, dst):
            partition_volumes[v.stage_id] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
    else:
        gain_function = calculate_stage_time_gain

        def update_function(v, dst):
            update_stage_times(v, dst, node_weights, edge_weights,
                               partition_volumes)

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

                gain = gain_function(n, dst, state)
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
                              rounds: int = 1,
                              objective: Objective = Objective.EDGE_CUT):

    # track which nodes belong to each partiiton
    partitions = {i: set() for i in partition_volumes}
    for n in node_weights:
        partitions[n.stage_id].add(n)

    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain

        def update_function(v, dst):
            partition_volumes[v.stage_id] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
            partitions[v.stage_id].discard(v)
            partitions[dst].add(v)
    else:
        gain_function = calculate_stage_time_gain

        def update_function(v, dst):
            update_stage_times(v, dst, node_weights, edge_weights,
                               partition_volumes)

            partitions[v.stage_id].discard(v)
            partitions[dst].add(v)

    state = PartitionState(edge_weights, node_weights, partition_volumes,
                           L_max)

    best_objective = calculate_edge_cut(node_weights.keys(), edge_weights)
    if objective is Objective.STAGE_TIME:
        best_objective = DoublePriority(max(partition_volumes.values()),
                                        best_objective)

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
                    candidate_moves.push_task(gain_function(node, B, state),
                                              (node, B))
            for node in partitions[B]:
                if all(i.stage_id <= A for i in node.in_edges):
                    candidate_moves.push_task(gain_function(node, A, state),
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

                gain = gain_function(node, dst, state)
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
                            gain = gain_function(i, B, state)
                            candidate_moves.push_task(gain, (i, B))
                else:
                    for o in node.out_edges:
                        if o.stage_id == B and all(i.stage_id <= A
                                               for i in o.in_edges):
                            gain = gain_function(o, A, state)
                            candidate_moves.push_task(gain, (o, A))

            # end of inner pass revert partition to best partition
            for n, dst in moves_to_best.items():
                update_function(n, dst)
                n.stage_id = dst


###################################################################################################

PartitionState = namedtuple(
    "PartitionState", "edge_weights node_weights partition_volumes L_max")

# assumes W(u,v) > 0
###################################################################################################

def calculate_stage_time_gain(v: SimpleNode, dest: int,
                              state: PartitionState) -> DoublePriority:
    # TODO maybe include summed distance from avg as penalty
    # or it's negation as gain
    # strict improvement in slowest stage time is too strict
    # most of moves will have a zero gain

    #TODO updating stage time is expensive O(d(v))
    # think about how to improve amortized complexity

    # create copy to not alter the original
    tmp = dict(state.partition_volumes)

    cur_max = max(tmp.values())

    edge_gain = update_stage_times(v, dest, state.node_weights,
                                   state.edge_weights, tmp)

    new_max = max(tmp.values())

    stage_gain = cur_max - new_max

    return DoublePriority(stage_gain, edge_gain)


def update_stage_times(v: SimpleNode, dest: int, node_weights: Dict[SimpleNode,
                                                                    float],
                       edge_weights: Dict[Tuple[SimpleNode, SimpleNode],
                                          float],
                       stage_times: Dict[int, float]) -> float:
    stage_times[v.stage_id] -= node_weights[v]
    stage_times[dest] += node_weights[v]

    edge_gain = 0

    for u in chain(v.in_edges, v.out_edges):
        if u.id < v.id:
            w = edge_weights[(u, v)]
        else:
            w = edge_weights[(v, u)]

        # record destinations so we won't overcount comm
        # only once per destination stage
        comms = set()
        if u.stage_id == v.stage_id:
            # u and v were at same partition
            # move adds comm less gain
            if (v.stage_id, w, dest, w) in comms:
                continue
            comms.add((v.stage_id, w, dest, w))
            edge_gain -= w
        elif u.stage_id == dest:
            # u and v will be at same partition
            # move reduces comm more gain
            if (v.stage_id, -w, dest, -w) in comms:
                continue
            comms.add((v.stage_id, -w, dest, -w))
            edge_gain += w
        else:
            # u and v were and will be at different partitions
            # move comm from src to dst no gain
            if (v.stage_id, -w, dest, w) in comms:
                continue
            comms.add((v.stage_id, -w, dest, w))

        for p0, comm0, p1, comm1 in comms:
            stage_times[p0] += comm0
            stage_times[p1] += comm1

    return edge_gain


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


def calculate_edge_cut(
        nodes: Iterator[SimpleNode],
        edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float]) -> float:
    edge_cut = 0
    for n in nodes:
        stages = set()
        for o in n.out_edges:
            if (n.stage_id != o.stage_id) and (o.stage_id not in stages):
                stages.add(o.stage_id)
                edge_cut += edge_weights[(n, o)]
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

ALGORITHMS = {
    ALGORITHM.SIMPLE_MOVES: simple_moves,
    ALGORITHM.ADVANCED_MOVES: advanced_moves,
    ALGORITHM.GLOBAL_MOVES: global_moves,
    ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES: Fiduccia_Mattheyses_moves
}

Solution = namedtuple("Solution",
                      "partition edge_cut slowest_stage volumes algorithm")

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
) -> Tuple[Graph, float, Dict[int, float]]:
    worker_args = [
        dict(graph=graph.state(),
             k=k,
             meta_algorithm=meta_algorithm,
             algorithm=alg,
             epsilon=epsilon,
             node_weight_function=node_weight_function,
             edge_weight_function=edge_weight_function,
             rounds=rounds,
             allocated_seconds=allocated_seconds,
             objective=objective,
             use_layers_graph=use_layers_graph) for alg in ALGORITHM
    ]

    with Pool(len(worker_args)) as pool:
        results = pool.map(worker, worker_args)

    assert len(results) == len(worker_args)

    best_solution = Solution(None, np.inf, np.inf, None, None)

    all_solutions = []
    all_initial_solutions=[]
    
    for solution,worker_sols,init_sol in results:
        if objective is Objective.EDGE_CUT:
            if (solution.edge_cut < best_solution.edge_cut) or (
                (solution.edge_cut == best_solution.edge_cut) and
                (solution.slowest_stage < best_solution.slowest_stage)):
                best_solution = solution
        elif (solution.slowest_stage < best_solution.slowest_stage) or (
            (solution.slowest_stage == best_solution.slowest_stage) and
            (solution.edge_cut < best_solution.edge_cut)):
            best_solution = solution
        
        all_solutions.extend(worker_sols)
        all_initial_solutions.extend(init_sol)

    partition, edge_cut, slowest_stage, volumes, algorithm = best_solution

    if DEBUG:
        for sol in all_initial_solutions:
            assert len(sol) == len(all_initial_solutions[0])
            assert set(sol) == set(all_initial_solutions[0])

        for sol in all_solutions:
            assert isinstance(sol,dict)
            assert set(sol.keys()) == set(range(len(graph)))

    for n in graph.nodes:
        n.stage_id = partition[n.id]

    assert graph.n_stages == k

    cutting_edges = 0
    cutting_scopes = []
    cutting_weights = defaultdict(list)
    for n in graph.nodes:
        stages=set()
        for u in n.out_edges:
            if (u.stage_id != n.stage_id) and (u.stage_id not in stages):
                stages.add(u.stage_id)
                cutting_edges += 1
                cutting_scopes.append(n.scope)
                cutting_weights[edge_weight_function(n,u)].append((n.scope,n.tensor_dtype))

    print()
    print("-I- Printing Partitioning Report")
    print(f"    allocated runtime: {allocated_seconds} seconds")
    print(f"    meta algorithm:{meta_algorithm.name}")
    print(f"    objective:{objective.name}")
    if objective is Objective.EDGE_CUT:
        print(f"    objective value: {edge_cut:.2f}")
    else:
        print(f"    objective value: {slowest_stage:.2f}")
    print(f"    graph size {len(graph)}")

    if DEBUG:
        print(f"    work graph size {len(all_initial_solutions[0])}")
        print(f"    total number of initial solutions {len(all_initial_solutions)}")
        print(f"    number of unique initial solutions {len(set(all_initial_solutions))}")
        print(f"    total number of solutions {len(all_solutions)}")
        print(f"    number of unique solutions {len(dedup_dicts(all_solutions))}")

    print(f"    best algorithm:{algorithm.name}")
    print(f"    number of cutting edges: {cutting_edges}")
    print(f"    edge cut:{edge_cut:.2f}")
    print(f"    volumes:{volumes}")

    if DEBUG:
        print("show edge cut:")
        for w,v in cutting_weights.items():
            print(f"w:{w}")
            for s,t in v:
                print(s,t)
            print()

    return graph, edge_cut, volumes


def worker(kwargs) -> Tuple[Solution,List[Dict[int,int]],List[Dict[int,int]]]:
    kwargs['graph'] = Graph(None, None, 
                        None, None, None).load_state(kwargs['graph']) 
    meta_algorithm = kwargs.pop("meta_algorithm")
    allocated_seconds = kwargs.pop("allocated_seconds")
    objective = kwargs['objective']    
    start = time.time()
    best_solution = Solution(None, np.inf, np.inf, None, None)
    steps = 0
    
    all_solutions = []
    all_initial_solutions = []

    while (time.time() - start) < allocated_seconds:
        seed = int.from_bytes(os.urandom(4), byteorder='little')
        random.seed(seed)
        
        if meta_algorithm is META_ALGORITH.SINGLE_LEVEL:
            init,solution, _, _ = single_level_partitioning(**kwargs)
        else:
            init,solution = multilevel_partitioning(**kwargs)

        if objective is Objective.EDGE_CUT:
            if (solution.edge_cut < best_solution.edge_cut) or (
                (solution.edge_cut == best_solution.edge_cut) and
                (solution.slowest_stage < best_solution.slowest_stage)):
                best_solution = solution
        elif (solution.slowest_stage < best_solution.slowest_stage) or (
            (solution.slowest_stage == best_solution.slowest_stage) and
            (solution.edge_cut < best_solution.edge_cut)):
            best_solution = solution

        if DEBUG:
            all_solutions.append(solution.partition)
            all_initial_solutions.append(init)
        
        steps += 1
    
    if DEBUG:
        alg_name = kwargs['algorithm'].name
        unique_init = set(all_initial_solutions)
        unique_solutions = len(dedup_dicts(all_solutions))
        solutions_str = f"unique_init {len(unique_init)} unique_solutions: {unique_solutions}"
        print(f"{alg_name} steps: {steps} {solutions_str}")

    return best_solution,all_solutions,all_initial_solutions


###################################################################################################


def single_level_partitioning(
    graph: Graph,
    algorithm: ALGORITHM = ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES,
    k: int = 4,
    epsilon: float = 0.1,
    node_weight_function: Optional[NodeWeightFunction] = None,
    edge_weight_function: Optional[EdgeWeightFunction] = None,
    objective: Objective = Objective.EDGE_CUT,
    rounds: int = 10,
    use_layers_graph=True
) -> Tuple[Dict[int,int],Solution, Dict[SimpleNode, float], Dict[Tuple[SimpleNode,
                                                         SimpleNode], float]]:
    if not use_layers_graph:
        work_graph = graph
    else:
        work_graph, layers_to_original = graph.layers_graph()

    if node_weight_function is None:
        node_weight_function = lambda n: 1

    if edge_weight_function is None:
        edge_weight_function = lambda u, v: 1

    node_weights = dict()
    edge_weights = dict()

    for n in work_graph.nodes:
        node_weights[n] = node_weight_function(n)
        for o in n.out_edges:
            edge_weights[(n, o)] = edge_weight_function(n, o)

    init_sol = initial_divide(work_graph, k, node_weights)

    partition_volumes = calculate_partition_volumes(k, node_weights)
    L_max = (1 + epsilon) * math.ceil(sum(partition_volumes.values()) / k)

    
    msg = "\n".join([
        "-I- balanced partitioning is not possible",
        f"   max allowed weight: {L_max:.2f}",
        f"   max node weight: {max(node_weights.values()):.2f}"
    ])

    assert all((v <= L_max for v in node_weights.values())), msg
    try:
        # NOTE we optimize the more stable edge cut objective
        # optimizing stage times directly is unstable and can create less partitions than requested
        # when selecting the best solution it could be according to the stage time objective 
        ALGORITHMS[algorithm](partition_volumes,
                            edge_weights,
                            node_weights,
                            L_max,
                            rounds=rounds,
                            objective=Objective.EDGE_CUT)

        #refine partition in a greedy fashion
        global_moves(partition_volumes,
                    edge_weights,
                    node_weights,
                    L_max,
                    rounds=1,
                    objective=Objective.EDGE_CUT)
    except Exception as e:
        print(f"error {algorithm.name}")
        work_graph.save_as_pdf(f"error_{algorithm.name}",".")
        raise e

    # induce partition from the layers graph to the original graph
    # recalculate partition metrics
    if use_layers_graph:
        graph.induce_layer_partition(work_graph, layers_to_original)
        #calculate metrics on original graph
        node_weights = dict()
        edge_weights = dict()
        for n in graph.nodes:
            node_weights[n] = node_weight_function(n)
            for o in n.out_edges:
                edge_weights[(n, o)] = edge_weight_function(n, o)

    #calculate metrics
    if objective is Objective.EDGE_CUT:
        partition_volumes = calculate_partition_volumes(k, node_weights)
    else:
        partition_volumes = calculate_stage_times(node_weights, edge_weights)
    edge_cut = calculate_edge_cut(graph.nodes, edge_weights)

    if DEBUG:
        assert graph.n_stages == k
        QuotientGraph(graph.nodes).selfcheck()

    return init_sol,Solution({n.id: n.stage_id
                     for n in graph.nodes}, edge_cut,
                    max(partition_volumes.values()), partition_volumes,
                    algorithm), node_weights, edge_weights


def multilevel_partitioning(
        graph: Graph,
        algorithm: ALGORITHM = ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES,
        k: int = 4,
        epsilon: float = 0.1,
        node_weight_function: Optional[NodeWeightFunction] = None,
        edge_weight_function: Optional[EdgeWeightFunction] = None,
        objective: Objective = Objective.EDGE_CUT,
        rounds: int = 10,
        use_layers_graph=True) -> Tuple[Dict[int,int],Solution]:

    # NOTE we optimize the more stable edge cut objective
    # optimizing stage times directly is unstable and can create less partitions than requested
    # when selecting the best solution it could be according to the stage time objective 
    initial_solution,single_level_solution, node_weights, edge_weights = single_level_partitioning(
        graph,
        algorithm=algorithm,
        k=k,
        epsilon=epsilon,
        node_weight_function=node_weight_function,
        edge_weight_function=edge_weight_function,
        objective=Objective.EDGE_CUT,
        rounds=rounds,
        use_layers_graph=use_layers_graph)
    partition_volumes = single_level_solution.volumes
    
    L_max = (1 + epsilon) * math.ceil(sum(partition_volumes.values()) / k)


    hierarchy = coarsening(graph, node_weights, edge_weights)
    # iterate in reverse order to coarsening
    # from smallest graph to largest graph
    for fine_graph, matching, coarse_graph in reversed(hierarchy):
        ALGORITHMS[algorithm](partition_volumes,
                              coarse_graph._edge_weights,
                              coarse_graph._node_weights,
                              L_max,
                              rounds=rounds,
                              objective=Objective.EDGE_CUT)
        refine(fine_graph, coarse_graph, matching)

    #update original graph
    root = hierarchy[0][0]
    for i in range(len(graph)):
        graph[i].stage_id = root[i].stage_id

    #calculate metrics
    if objective is Objective.EDGE_CUT:
        partition_volumes = calculate_partition_volumes(k, node_weights)
    else:
        partition_volumes = calculate_stage_times(node_weights, edge_weights)
    edge_cut = calculate_edge_cut(graph.nodes, edge_weights)

    return initial_solution,Solution({n.id: n.stage_id
                     for n in graph.nodes}, edge_cut,
                    max(partition_volumes.values()), partition_volumes,
                    algorithm)


###################################################################################################


def dedup_dicts(items: List[dict])->List[dict]:
    dedupped = [json.loads(i) for i in set(json.dumps(item, sort_keys=True) for item in items)]
    return [{int(k):v for k,v in d.items()} for d in dedupped]


def all_topo(graph):
    # Traverse adjacency lists to fill indegrees of 
    # vertices.  This step takes O(V+E) time 
    in_degree = {n:len(n.in_edges) for n in graph.nodes}

    # Create an queue and enqueue all vertices with 
    # indegree 0 
    queue = []
    for n,deg in in_degree.items():
        if deg == 0:
            queue.append(n)

    yield from process_queue(len(graph),queue, in_degree, [], 0)


def process_queue(n, queue, in_degree, top_order, cnt):
    if queue:
        # We have multiple possible next nodes, generate all possbile variations
        for u in queue:

            # create temp copies for passing to process_queue
            curr_top_order = top_order + [u]
            curr_in_degree = dict(in_degree)
            curr_queue = list(queue)
            curr_queue.remove(u)

            # Iterate through all neighbouring nodes 
            # of dequeued node u and decrease their in-degree 
            # by 1 
            for i in u.out_edges:
                curr_in_degree[i] -= 1
                # If in-degree becomes zero, add it to queue 
                if curr_in_degree[i] == 0:
                    curr_queue.append(i)

            yield from process_queue(n,curr_queue, curr_in_degree, curr_top_order, cnt + 1)  # continue recursive

    elif cnt != n:
        print("There exists a cycle in the graph")
    else:
        sol=tuple([n.id for n in top_order])
        yield sol


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))