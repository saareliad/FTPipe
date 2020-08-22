from torch.nn.parameter import Parameter
from pytorch_Gpipe.utils import layerDict, tensorDict
from torch.nn.modules.module import Module
from pytorch_Gpipe.model_profiling import Graph, NodeWeightFunction, EdgeWeightFunction
from .data_structures import QuotientGraph, SimpleNode, PriorityQueue, VerticeStageConnections
from .gpa import coarsening, refine
import random
import math
import numpy as np
from typing import Tuple, Dict, Optional,NamedTuple
from collections import defaultdict
import enum
from multiprocessing import Pool
import time
import os

###################################################################################################

# TODO add maximum suppression hueristic aka minimize objective for the stage which has the largest objective value


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

    def __repr__(self) -> str:
        return self.name


class Objective(enum.Enum):
    EDGE_CUT = 1
    STAGE_TIME = 2

    def __repr__(self) -> str:
        return self.name


class Constraint(enum.Enum):
    MEMORY = 1
    TIME = 2

    def __repr__(self) -> str:
        return self.name


###################################################################################################

#create initial partitioning by taking consecutive blocks of equal weight
#the blocks are consecutive blocks of nodes acquired by a Khan's algorithm
def initial_divide(graph: Graph, k: int,
                   weights: Dict[SimpleNode, float])->Tuple[int,...]:
    random_topo_sort = random_Khan_algorithm(graph)
    weights = np.asarray([weights[n] for n in random_topo_sort])
    cumulative_weights = np.cumsum(weights)

    total_weight = cumulative_weights[-1]
    avg_weight = total_weight / k

    Vs = []

    options = [math.floor(avg_weight), math.ceil(avg_weight)]
    acc = 0
    # k partitions require k-1 seperators
    while len(Vs) < k - 1:
        stage_weight = options[random.randint(0, 1)]
        acc += stage_weight
        Vs.append(np.searchsorted(cumulative_weights, acc))

    idxs = [-1] + Vs + [len(cumulative_weights) - 1]

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
def simple_moves(constraint: Constraint,
                 objective: Objective,
                 stage_volumes: Dict[int, float],
                 params_per_stage: Dict[int,float],
                 edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                 node_weights: Dict[SimpleNode, float],
                 params_per_node: Dict[SimpleNode,float],
                 L_max: float,
                 rounds: int):
    connections = VerticeStageConnections(node_weights)

    def update_function(v:SimpleNode, dst:int):
        stage_volumes[v.stage_id] -= node_weights[v]
        stage_volumes[dst] += node_weights[v]

        params_per_stage[v.stage_id] -= params_per_node[v]
        params_per_stage[dst] += params_per_node[v]

        connections.move_node(v, dst)
        v.stage_id = dst


    satisfies_constraint = CONSTRAINTS[constraint]
    gain_function = GAINS[objective]

    state = PartitionState(stage_volumes,params_per_stage,node_weights,edge_weights,params_per_node,L_max)

    # we use 0 based indexing
    k = len(stage_volumes) - 1

    nodes = list(node_weights.keys())
    for _ in range(rounds):
        changed = False

        random.shuffle(nodes)
        # O(E)
        for n in nodes:
            gain_left = -np.inf
            if (n.stage_id > 0
                ) and (not connections.has_in_connection(n, n.stage_id)) and satisfies_constraint(n,n.stage_id-1,state):
                gain_left = gain_function(n, n.stage_id - 1, state)

            gain_right = -np.inf
            if (n.stage_id < k
                ) and (not connections.has_out_connection(n, n.stage_id)) and satisfies_constraint(n,n.stage_id+1,state):
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

        if not changed:
            break


# move nodes between all partitions as long as edge cut improves and constraints are enforced
# uses a sufficient condition for enforcing acyclicicity not a necessary condition
# as such some options are skipped
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def advanced_moves(constraint: Constraint,
                 objective: Objective,
                 stage_volumes: Dict[int, float],
                 params_per_stage: Dict[int,float],
                 edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                 node_weights: Dict[SimpleNode, float],
                 params_per_node: Dict[SimpleNode,float],
                 L_max: float,
                 rounds: int):

    def update_function(v:SimpleNode, dst:int):
        stage_volumes[v.stage_id] -= node_weights[v]
        stage_volumes[dst] += node_weights[v]

        params_per_stage[v.stage_id] -= params_per_node[v]
        params_per_stage[dst] += params_per_node[v]

        v.stage_id = dst

    satisfies_constraint = CONSTRAINTS[constraint]
    gain_function = GAINS[objective]


    state = PartitionState(stage_volumes,params_per_stage,node_weights,edge_weights,params_per_node,L_max)

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
                gain = -np.inf
                if satisfies_constraint(n,dst,state):
                    gain = gain_function(n, dst, state)
                moves[gain].append(dst)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue

            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves, 1)[0]

            update_function(n, dst)

        if not changed:
            break


# move nodes between all partitions as long as edge cut improves and constraints are enforced
# uses Khan's algorithm to ensure we do not create cycles in the quotient graph
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def global_moves(constraint: Constraint,
                 objective: Objective,
                 stage_volumes: Dict[int, float],
                 params_per_stage: Dict[int,float],
                 edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                 node_weights: Dict[SimpleNode, float],
                 params_per_node: Dict[SimpleNode,float],
                 L_max: float,
                 rounds: int):

    def update_function(v:SimpleNode, dst:int):
        stage_volumes[v.stage_id] -= node_weights[v]
        stage_volumes[dst] += node_weights[v]

        params_per_stage[v.stage_id] -= params_per_node[v]
        params_per_stage[dst] += params_per_node[v]

    satisfies_constraint = CONSTRAINTS[constraint]
    gain_function = GAINS[objective]

    state = PartitionState(stage_volumes,params_per_stage,node_weights,edge_weights,params_per_node,L_max)

    quotient_graph = QuotientGraph(node_weights.keys())
    nodes = list(node_weights.keys())

    for _ in range(rounds):
        changed = False
        random.shuffle(nodes)
        # O(E(k+mq))
        for n in nodes:
            moves = defaultdict(list)
            for dst in stage_volumes.keys():
                if dst == n.stage_id:
                    continue

                gain = -np.inf
                if satisfies_constraint(n,dst,state) and (not quotient_graph.move_creates_cycle(n, dst)):
                    gain = gain_function(n, dst, state)

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
def Fiduccia_Mattheyses_moves(constraint: Constraint,
                              objective: Objective,
                              stage_volumes: Dict[int, float],
                              params_per_stage: Dict[int,float],
                              edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float],
                              node_weights: Dict[SimpleNode, float],
                              params_per_node: Dict[SimpleNode,float],
                              L_max: float,
                              rounds: int):

    # track which nodes belong to each partiiton
    partitions = {i: set() for i in stage_volumes}
    for n in node_weights:
        partitions[n.stage_id].add(n)

    def update_function(v:SimpleNode, dst:int):
        stage_volumes[v.stage_id] -= node_weights[v]
        stage_volumes[dst] += node_weights[v]

        partitions[v.stage_id].discard(v)
        partitions[dst].add(v)

        params_per_stage[v.stage_id] -= params_per_node[v]
        params_per_stage[dst] += params_per_node[v]

        v.stage_id = dst

    satisfies_constraint = CONSTRAINTS[constraint]
    gain_function = GAINS[objective]

    state = PartitionState(stage_volumes,params_per_stage,node_weights,edge_weights,params_per_node,L_max)

    best_objective = calculate_edge_cut(edge_weights)

    all_blocks = list(stage_volumes.keys())

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
                elif not satisfies_constraint(node,dst,state):
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


HEURISTICS = {
    ALGORITHM.SIMPLE_MOVES: simple_moves,
    ALGORITHM.ADVANCED_MOVES: advanced_moves,
    ALGORITHM.GLOBAL_MOVES: global_moves,
    ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES: Fiduccia_Mattheyses_moves
}



###################################################################################################


class PartitionState(NamedTuple):
    stage_volumes: Dict[int,float]
    params_per_stage: Dict[int,float]
    node_weights: Dict[SimpleNode,float]
    edge_weights: Dict[Tuple[SimpleNode,SimpleNode],float]
    params_per_node: Dict[SimpleNode,float]
    L_max: float


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


def calculate_edge_cut(edge_weights: Dict[Tuple[SimpleNode, SimpleNode], float]) -> float:
    edge_cut = 0

    visited = set()
    for (u,v),w in edge_weights.items():
        if (u.id,v.stage_id) not in visited:
            edge_cut += w
            visited.add((u.id,v.stage_id))

    return edge_cut


def calculate_stage_volumes(node_weights: Dict[SimpleNode, float]) -> Dict[int, float]:
    stage_volumes = defaultdict(lambda : 0)
    for n, w in node_weights.items():
        stage_volumes[n.stage_id] += w

    return dict(stage_volumes)


def calculate_stage_time_gain(v:SimpleNode, dst:int, state:PartitionState) -> float:
    node_weights = state.node_weights
    edge_weights = state.edge_weights
    volumes = state.stage_volumes


    prev_max = max(volumes[v.stage_id],volumes[dst])

    new_max = max(volumes[v.stage_id]-node_weights[v],volumes[dst]+node_weights[v])

    delta = prev_max - new_max

    # TODO decide wether to include communication
    
    return delta


def calculate_stage_times(node_weights: Dict[SimpleNode, float],
                          edge_weights: Dict[Tuple[SimpleNode, SimpleNode],float]) -> Dict[int, float]:
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
            # TODO decide wether to include communication
            # stage_times[n.stage_id] += e
            # stage_times[o.stage_id] += e

    return dict(stage_times)


def calculate_params_per_node(model:Module,graph:Graph)->Dict[int,float]:
    layers = layerDict(model,graph.depth,graph.basic_blocks)
    tensors = tensorDict(model)

    params_per_node=dict()

    for n in graph.nodes:
        if n.scope in layers:
            params_per_node[n.id] = sum(t.numel() for t in layers[n.scope].parameters())
        elif (n.value_type is Parameter) and (n.scope in tensors):
            params_per_node[n.id] = tensors[n.scope].numel()
        else:
            params_per_node[n.id] = 0
    
    return params_per_node


def calculate_params_per_stage(params_per_node:Dict[SimpleNode,float])->Dict[int,float]:
    params_per_stage = defaultdict(lambda:0)

    for n,p in params_per_node.items():
        params_per_stage[n.stage_id] += p
    
    return dict(params_per_stage)


GAINS = {Objective.EDGE_CUT:calculate_edge_gain,
        Objective.STAGE_TIME:calculate_stage_time_gain}


###################################################################################################

def move_satisfies_time_constraint(v:SimpleNode,dst:int,state:PartitionState)->bool:
    node_weights = state.node_weights
    volumes = state.stage_volumes

    return (volumes[dst] + node_weights[v]) < state.L_max


def move_satisifies_memory_constraint(v:SimpleNode,dst:int,state:PartitionState)->bool:
    params_per_node = state.params_per_node
    params_per_stage = state.params_per_stage

    return (params_per_stage[dst] + params_per_node[v]) < state.L_max


CONSTRAINTS = { Constraint.TIME:move_satisfies_time_constraint,
                Constraint.MEMORY:move_satisifies_memory_constraint}

###################################################################################################


def acyclic_partition(
        model:Module,
        graph: Graph,
        k: int,
        epsilon: float = 0.1,
        node_weight_function: Optional[NodeWeightFunction] = None,
        edge_weight_function: Optional[EdgeWeightFunction] = None,
        constraint: Constraint = Constraint.TIME,
        objective: Objective = Objective.EDGE_CUT,
        meta_algorithm: META_ALGORITH = META_ALGORITH.SINGLE_LEVEL,
        maximum_constraint_value: Optional[float] = None,
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

    params_per_node = calculate_params_per_node(model,work_graph)

    worker_args = [
        dict(graph=work_graph._remove_parallel_edges().state(),
             params_per_node=params_per_node,
             k=k,
             meta_algorithm=meta_algorithm,
             algorithm=alg,
             epsilon=epsilon,
             node_weight_function=node_weight_function,
             edge_weight_function=edge_weight_function,
             rounds=rounds,
             allocated_seconds=allocated_seconds,
             objective=objective,
             constraint=constraint,
             maximum_constraint_value=maximum_constraint_value) for alg in ALGORITHM
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
    params_per_node = dict(kwargs["params_per_node"])
    for u in graph.nodes:
        node_weights[u] = nwf(u)
        params_per_node[u] = params_per_node.pop(u.id)
        for o in u.out_edges:
            edge_weights[(u,o)] = ewf(u,o)

    kwargs['params_per_node'] = params_per_node
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


def single_level_partitioning(graph: Graph,
                                node_weights: Dict[SimpleNode,float],
                                edge_weights: Dict[Tuple[SimpleNode,SimpleNode],float],
                                params_per_node: Dict[SimpleNode,float],
                                algorithm: ALGORITHM,
                                k: int,
                                epsilon: float,
                                constraint: Constraint,
                                maximum_constraint_value: Optional[float],
                                objective: Objective,
                                rounds: int) -> Tuple[Dict[int,int],float,float]:
                                
    if constraint is Constraint.TIME:
        constraint_weights = node_weights
    else:
        constraint_weights = params_per_node
    
    initial_divide(graph,k,constraint_weights)

    #TODO if we decide to include comm for stage_time objective we'll need to address this here
    stage_volumes = calculate_stage_volumes(node_weights)
    params_per_stage = calculate_params_per_stage(params_per_node)

    if constraint is Constraint.TIME:
        constraint_per_stage = stage_volumes
    else:
        constraint_per_stage = params_per_stage

    avg_constraint_value = (sum(constraint_per_stage.values()) / k)

    if maximum_constraint_value is None:
        L_max = (1 + epsilon) * math.ceil(sum(constraint_per_stage.values()) / k)
    else:
        L_max = maximum_constraint_value
    
    msg = "\n".join([
        f"-I- partitioning with {constraint.name} constraint is not possible",
        f"   max allowed stage constraint: {L_max:.2f}",
        f"   average constraint value: {avg_constraint_value:.2f}"
    ])

    assert avg_constraint_value < L_max, msg

    # NOTE we optimize the more stable edge cut objective
    # optimizing stage times directly is unstable and can create less partitions than requested
    # when selecting the best solution it could be according to the stage time objective 
    HEURISTICS[algorithm](constraint,
                        objective,
                        stage_volumes,
                        params_per_stage,
                        edge_weights,
                        node_weights,
                        params_per_node,
                        L_max,
                        rounds)

    #refine partition in a greedy fashion
    global_moves(constraint,
                objective,
                stage_volumes,
                params_per_stage,
                edge_weights,
                node_weights,
                params_per_node,
                L_max,
                rounds=1)

    if objective is Objective.STAGE_TIME:
        stage_volumes = calculate_stage_times(node_weights,edge_weights)

    edge_cut = calculate_edge_cut(edge_weights)

    return {n.id:n.stage_id for n in graph.nodes},edge_cut,max(stage_volumes.values())


def multilevel_partitioning(
                        graph: Graph,
                        node_weights: Dict[SimpleNode,float],
                        edge_weights: Dict[Tuple[SimpleNode,SimpleNode],float],
                        params_per_node: Dict[SimpleNode,float],
                        algorithm: ALGORITHM,
                        k: int,
                        epsilon: float,
                        constraint:Constraint,
                        maximum_constraint_value: Optional[float],
                        objective: Objective,
                        rounds: int
                        ) -> Tuple[Dict[int,int],float,float]:

    single_level_partitioning(graph,
                                params_per_node=params_per_node,
                                node_weights=node_weights,
                                edge_weights=edge_weights,
                                algorithm=algorithm,
                                k=k,
                                epsilon=epsilon,
                                constraint=constraint,
                                maximum_constraint_value=maximum_constraint_value,
                                objective=objective,
                                rounds=rounds)
    
    #TODO if we decide to include comm for stage_time objective we'll need to address this here
    stage_volumes = calculate_stage_volumes(node_weights)
    params_per_stage = calculate_params_per_stage(params_per_node)

    if constraint is Constraint.TIME:
        constraint_per_stage = stage_volumes
    else:
        constraint_per_stage = params_per_stage

    if maximum_constraint_value is None:
        L_max = (1 + epsilon) * math.ceil(sum(constraint_per_stage.values()) / k)
    else:
        L_max = maximum_constraint_value
    
    hierarchy = coarsening(graph, node_weights, edge_weights,params_per_node)
    # iterate in reverse order to coarsening
    # from smallest graph to largest graph
    for fine_graph, matching, coarse_graph in reversed(hierarchy):
        HEURISTICS[algorithm](constraint,
                              objective,
                              stage_volumes,
                              params_per_stage,
                              coarse_graph._edge_weights,
                              coarse_graph._node_weights,
                              coarse_graph._params_per_node,
                              L_max,
                              rounds)
        refine(fine_graph, coarse_graph, matching)

    #update original graph
    root = hierarchy[0][0]
    for i in range(len(graph)):
        graph[i].stage_id = root[i].stage_id
    
    if objective is Objective.STAGE_TIME:
        stage_volumes = calculate_stage_times(node_weights,edge_weights)

    edge_cut = calculate_edge_cut(edge_weights)

    return {n.id:n.stage_id for n in graph.nodes},edge_cut,max(stage_volumes.values())


###################################################################################################


class DefaultWeightFunction():
    def __call__(self,u:SimpleNode)->float:
        return 1

class DefaultEdgeWeightFunction():
    def __call__(self,u:SimpleNode,v:SimpleNode)->float:
        return 1