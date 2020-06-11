from pytorch_Gpipe.model_profiling import Graph,Node,NodeWeightFunction,EdgeWeightFunction
from .data_structures import QuotientGraph,PartitionNode,PriorityQueue,DoublePriority
from ..METIS_partitioning.partition_graph import induce_layer_partition
import random
import math
import numpy as np
from typing import Tuple,Dict,Optional,Iterator
from collections import defaultdict,namedtuple
from itertools import chain
import enum
from multiprocessing import Pool
import time

DEBUG = False


class ALGORITHM(enum.Enum):
    SIMPLE_MOVES=1
    ADVANCED_MOVES=2
    GLOBAL_MOVES=3
    FIDUCCIA_MATTHEYSES_MOVES=4

    def __repr__(self):
        return self.name

class Objective(enum.Enum):
    EDGE_CUT=1
    STAGE_TIME=2

    def __repr__(self):
        return self.name


#create initial partitioning by taking consecutive blocks of equal weight
#the blocks are consecutive blocks of nodes acquired by a Khan's algorithm
def initial_divide(graph:Graph,k:int,node_weights:Dict[Node,float])->QuotientGraph:
    random_topo_sort=random_Khan_algorithm(graph)
    node_weights = np.asarray([node_weights[n] for n in random_topo_sort])
    cumulative_node_weights = np.cumsum(node_weights)

    total_weight = cumulative_node_weights[-1]
    avg_weight = total_weight/k

    Vs=[]

    options = [math.floor(avg_weight),math.ceil(avg_weight)]
    acc = 0
    #k partitions require k-1 seperators
    while len(Vs) < k-1:
        stage_weight = options[random.randint(0,1)]
        acc+=stage_weight
        Vs.append(np.searchsorted(cumulative_node_weights,acc))

    idxs=[-1]+Vs+[len(cumulative_node_weights)-1]

    idxs=list(zip(map(lambda i: i+1,idxs),idxs[1:]))
    
    #set partitioning
    for i,(start,end) in enumerate(idxs):
        for n in random_topo_sort[start:end+1]:
            n.part=i

    for n in graph.nodes:
        for o in n.out_edges:
            assert n.part <=o.part

    return QuotientGraph(graph.nodes)


# create a random topological sorting
def random_Khan_algorithm(graph:Graph):
    S=[]
    T=[]

    degs=dict()
    # o(V)
    for n in graph.nodes:
        if len(n.in_edges)==0:
            S.append(n)
        else:
            degs[n]=len(n.in_edges)
    # O(E)
    while S:
        idx = random.randint(0,len(S)-1)
        n=S[idx]
        del S[idx]
        T.append(n)
        for o in n.out_edges:
            degs[o]-=1
            if degs[o] == 0:
                S.append(o)


    #if we have a cycle S will be empty and T will not contain all of the nodes

    idxs = dict(zip(T,range(len(T))))

    for n in T:
        n_idx = idxs[n]
        assert all(idxs[o] > n_idx for o in n.out_edges)

    return T


#move nodes between adjacent partitions as long as edge cut improves and constraints are enforced
# aka i=>i+1 or i=>i+1
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def simple_moves(partition_volumes:Dict[int,float],
                edge_weights:Dict[Tuple[Node,Node],float],
                node_weights:Dict[Node,float],
                L_max:float,
                rounds:int=1,
                objective:Objective = Objective.EDGE_CUT):
    
    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain
        def update_function(v,dst):
            partition_volumes[v.part] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
    else:
        gain_function = calculate_stage_time_gain
        def update_function(v,dst):
            update_stage_times(v,dst,node_weights,edge_weights,partition_volumes)


    state = PartitionState(edge_weights,node_weights,partition_volumes,L_max)

    #we use 0 based indexing
    k=len(partition_volumes)-1
    
    for _ in range(rounds):
        changed=False

        # O(E)
        for n in node_weights.keys():
            edge_gain_left = -np.inf
            if (n.part > 0) and (C_in(n,n.part,edge_weights)==0) and ((partition_volumes[n.part-1]+node_weights[n])<L_max):
                edge_gain_left = gain_function(n,n.part-1,state)
            
            edge_gain_right = -np.inf
            if (n.part < k) and (C_out(n,n.part,edge_weights)==0) and ((partition_volumes[n.part+1]+node_weights[n])<L_max):
                edge_gain_right = gain_function(n,n.part+1,state)
            
            moves = defaultdict(list)
            moves[edge_gain_left].append(n.part-1)
            moves[edge_gain_right].append(n.part+1)
            
            max_gain = max(moves.keys())
            if max_gain < 0:
                continue

            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves,1)[0]
            
            update_function(n,dst)
            n.part = dst

        if not changed:
            break


#move nodes between all partitions as long as edge cut improves and constraints are enforced
#uses a sufficient condition for enforcing acyclicicity not a necessary condition
# as such some options are skipped
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def advanced_moves(partition_volumes:Dict[int,float],
                edge_weights:Dict[Tuple[Node,Node],float],
                node_weights:Dict[Node,float],
                L_max:float,
                rounds:int=1,
                objective:Objective = Objective.EDGE_CUT):
    
    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain
        def update_function(v,dst):
            partition_volumes[v.part] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
    else:
        gain_function = calculate_stage_time_gain
        def update_function(v,dst):
            update_stage_times(v,dst,node_weights,edge_weights,partition_volumes)

    state = PartitionState(edge_weights,node_weights,partition_volumes,L_max)

    for _ in range(rounds):
        changed=False

        # O(E)
        for n in node_weights.keys():
            # [A,B] is the eligible partition range that n can be placed in
            A = max((i.part for i in n.in_edges),default=n.part)
            B = min((o.part for o in n.out_edges),default=n.part)
            
            if A==B:
                # n has in and out connections in it's partition
                # cannot be moved
                continue

            moves = defaultdict(list)
            for j in range(A,B+1):
                if j == n.part:
                    continue
                edge_gain = gain_function(n,j,state)
                if (partition_volumes[j]+node_weights[n]) > L_max:
                    edge_gain = -np.inf
                moves[edge_gain].append(j)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue
            
            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves,1)[0]
            
            update_function(n,dst)
            n.part = dst

        if not changed:
            break


#move nodes between all partitions as long as edge cut improves and constraints are enforced
#uses Khan's algorithm to ensure we do not create cycles in the quotient graph
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def global_moves(partition_volumes:Dict[int,float],
                edge_weights:Dict[Tuple[Node,Node],float],
                node_weights:Dict[Node,float],
                L_max:float,
                rounds:int=1,
                objective:Objective = Objective.EDGE_CUT):
    
    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain
        def update_function(v,dst):
            partition_volumes[v.part] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
    else:
        gain_function = calculate_stage_time_gain
        def update_function(v,dst):
            update_stage_times(v,dst,node_weights,edge_weights,partition_volumes)

    state = PartitionState(edge_weights,node_weights,partition_volumes,L_max)
    
    quotient_graph = QuotientGraph(node_weights.keys())
    for _ in range(rounds):
        changed=False

        # O(E(k+mq))
        for n in node_weights.keys():
            moves = defaultdict(list)
            for j in range(len(partition_volumes)):
                if j == n.part:
                    continue

                edge_gain = gain_function(n,j,state)
                if ((partition_volumes[j]+node_weights[n]) > L_max) or quotient_graph.move_creates_cycle(n,j):
                    edge_gain = -np.inf
        
                moves[edge_gain].append(j)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue
            
            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves,1)[0]

            update_function(n,dst)
            quotient_graph.move_node(n,dst)

            if DEBUG:
                quotient_graph.selfcheck()

        if not changed:
            break


# move nodes between partitions
# moves with negative gain are also eligible in order to escape local minima
# the partitioning with the best objective will be returned
def Fiduccia_Mattheyses_moves(partition_volumes:Dict[int,float],
                            edge_weights:Dict[Tuple[Node,Node],float],
                            node_weights:Dict[Node,float],
                            L_max:float,
                            rounds:int=1,
                            objective:Objective = Objective.EDGE_CUT):
    
    if objective is Objective.EDGE_CUT:
        gain_function = calculate_edge_gain
        def update_function(v,dst):
            partition_volumes[v.part] -= node_weights[v]
            partition_volumes[dst] += node_weights[v]
    else:
        gain_function = calculate_stage_time_gain
        def update_function(v,dst):
            update_stage_times(v,dst,node_weights,edge_weights,partition_volumes)

    state = PartitionState(edge_weights,node_weights,partition_volumes,L_max)
    
    quotient_graph = QuotientGraph(node_weights.keys())
    best_objective = calculate_edge_cut(node_weights.keys(),edge_weights)
    if objective is Objective.STAGE_TIME:
        best_objective = DoublePriority(max(partition_volumes.values()),best_objective)

    all_blocks = list(partition_volumes.keys())

    for _ in range(rounds):
        active_blocks = set(all_blocks)

        #outer pass
        while active_blocks:
            #select A and B such that A < B and at least one of A,B is active
            A,B = random.sample(all_blocks,2)
            if A > B:
                A,B = B,A
            if A == B or not (A in active_blocks or B in active_blocks):
                continue

            active_blocks.discard(A)
            active_blocks.discard(B)

            candidate_moves = PriorityQueue()
            for node in quotient_graph[A]:
                if all(o.part >= B for o in node.out_edges):
                    candidate_moves.push_task(gain_function(node,B,state),(node,B))
            for node in quotient_graph[B]:
                if all(i.part <= A for i in node.in_edges):
                    candidate_moves.push_task(gain_function(node,A,state),(node,A))
            
            locked_nodes = set()
            moves_to_best = dict()
            current_objective = best_objective
            
            #inner pass
            while len(candidate_moves) > 0:
                node,dst = candidate_moves.pop_task()

                #check if the move is still valid
                if node in locked_nodes:
                    continue
                elif (partition_volumes[dst] + node_weights[node]) > L_max:
                    continue
                elif (node.part == A) and any(o.part < B for o in node.out_edges):
                    continue
                elif(node.part == B) and any(i.part > A for i in node.in_edges):
                    continue
                
                locked_nodes.add(node)

                edge_gain = gain_function(node,dst,state)
                current_objective -= edge_gain
                src = node.part

                update_function(node,dst)
                quotient_graph.move_node(node,dst)

                if DEBUG:
                    quotient_graph.selfcheck()

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
                        if i.part == A and all(o.part >= B for o in i.out_edges):
                            edge_gain = gain_function(i,B,state)
                            candidate_moves.push_task(edge_gain,(i,B))
                else:
                    for o in node.out_edges:
                        if o.part == B and all(i.part <= A for i in o.in_edges):
                            edge_gain = gain_function(o,A,state)
                            candidate_moves.push_task(edge_gain,(o,A))

            #end of inner pass revert partition to best partition
            for n,dst in moves_to_best.items():
                update_function(n,dst)
                quotient_graph.move_node(n,dst)

            if DEBUG:
                quotient_graph.selfcheck()




PartitionState = namedtuple("PartitionState","edge_weights node_weights partition_volumes L_max")


#assumes W(u,v) > 0

def calculate_stage_time_gain(v:Node,dest:int,state:PartitionState)->DoublePriority:
    #TODO maybe include summed distance from avg as penalty
    # or it's negation as gain
    # strict improvement in slowest stage time is too strict
    # most of moves will have a zero gain

    #TODO updating stage time is expensive O(d(v))
    # think about how to improve amortized complexity
    

    #create copy to not alter the original
    tmp = dict(state.partition_volumes)

    cur_max = max(tmp.values())

    edge_gain = update_stage_times(v,dest,state.node_weights,state.edge_weights,tmp)

    new_max = max(tmp.values())

    stage_gain = cur_max - new_max

    return DoublePriority(stage_gain,edge_gain)


def update_stage_times(v:Node,dest:int,node_weights:Dict[Node,float],
                                edge_weights:Dict[Tuple[Node,Node],float],stage_times:Dict[int,float])->float:
    #TODO right now we if we have u->v and u->w
    # and v,w are in the same partition
    # we count the comm twice
    
    stage_times[v.part] -= node_weights[v]
    stage_times[dest] += node_weights[v]

    edge_gain = 0

    for u in chain(v.in_edges,v.out_edges):
        if u.id < v.id:
            w = edge_weights[(u,v)]
        else:
            w = edge_weights[(v,u)]
        
        if u.part == v.part:
            # u and v were at same partition
            # move adds comm less gain
            stage_times[u.part] += w
            stage_times[dest] += w
            edge_gain -= w
        elif u.part == dest:
            # u and v will be at same partition
            # move reduces comm more gain
            stage_times[v.part] -= w
            stage_times[dest] -= w
            edge_gain += w
        else:
            # u and v were and will be at different partitions
            # move comm from src to dst no gain
            stage_times[v.part] -= w
            stage_times[dest] += w
    
    return edge_gain


def calculate_edge_gain(v:Node,dest:int,state:PartitionState)->float:
    # C_in(v,dest,edge_weights) - C_out(v,v.part,edge_weights) + C_out(v,dest,edge_weights) - C_in(v,v.part,edge_weights)
    # does not use C_in / C_out for performance reasons
    edge_weights = state.edge_weights
    gain = 0
    for n in v.in_edges:
        if n.part == dest:
            gain += edge_weights[(n,v)]
        elif n.part == v.part:
            gain -= edge_weights[(n,v)]

    for n in v.out_edges:
        if n.part == dest:
            gain += edge_weights[(v,n)]
        elif n.part == v.part:
            gain -= edge_weights[(v,n)]

    return gain


def C_in(v:Node,i:int,edge_weights:Dict[Tuple[Node,Node],float])->float:
    return sum(edge_weights[(u,v)] for u in v.in_edges if u.part == i)


def C_out(v:Node,i:int,edge_weights:Dict[Tuple[Node,Node],float])->float:
    return sum(edge_weights[(v,u)] for u in v.out_edges if u.part == i)


def calculate_edge_cut(nodes:Iterator[Node],edge_weights:Dict[Tuple[Node,Node],float])->float:
    edge_cut=0
    for n in nodes:
        for o in n.out_edges:
            if n.part != o.part:
                edge_cut += edge_weights[(n,o)]
    return edge_cut


def calculate_partition_volumes(k:int,node_weights:Dict[Node,float])->Dict[int,float]:
    partition_volumes={i:0 for i in range(k)}
    for n,w in node_weights.items():
        partition_volumes[n.part]+=w
    
    return partition_volumes


def calculate_stage_times(node_weights:Dict[Node,float],edge_weights:Dict[Tuple[Node,Node],float])->Dict[int,float]:
    #TODO right now we if we have u->v and u->w
    # and v,w are in the same partition
    # we count the comm twice
    stage_times = defaultdict(lambda : 0)

    for n,w in node_weights.items():
        stage_times[n.part]+=w
        for u in chain(n.in_edges,n.out_edges):
            if u.part < n.part:
                stage_times[n.part]+=edge_weights[(u,n)]
            elif u.part > n.part:
                stage_times[n.part]+=edge_weights[(n,u)]

    return dict(stage_times)


ALGORITHMS = {
    ALGORITHM.SIMPLE_MOVES:simple_moves,
    ALGORITHM.ADVANCED_MOVES:advanced_moves,
    ALGORITHM.GLOBAL_MOVES:global_moves,
    ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES:Fiduccia_Mattheyses_moves
}

Solution = namedtuple("Solution","partition edge_cut slowest_stage volumes algorithm")


def acyclic_partition(graph:Graph,k:int,epsilon:float=0.1,node_weight_function:Optional[NodeWeightFunction]=None,
                    edge_weight_function:Optional[EdgeWeightFunction]=None,objective:Objective=Objective.STAGE_TIME,rounds:int=10,allocated_seconds:int=20,use_layers_graph:bool=True)->Tuple[Graph,float,Dict[int,float]]:
    worker_args=[dict(graph = graph.state(),
                        k=k,
                        algorithm=alg,
                        epsilon=epsilon,
                        node_weight_function=node_weight_function,
                        edge_weight_function=edge_weight_function,
                        rounds=rounds,
                        allocated_seconds=allocated_seconds,
                        seed=random.randint(0,2**32),
                        objective = objective,
                        use_layers_graph=use_layers_graph) for alg in ALGORITHM]

    with Pool(len(worker_args)) as pool:
        results=pool.map(worker, worker_args)

    assert len(results) == len(ALGORITHMS)

    best_solution = Solution(None,np.inf,np.inf,None,None)
    for solution in results:
        if objective is Objective.EDGE_CUT:
            if (solution.edge_cut < best_solution.edge_cut) or ((solution.edge_cut==best_solution.edge_cut) and (solution.slowest_stage < best_solution.slowest_stage)):
                best_solution = solution
        elif (solution.slowest_stage < best_solution.slowest_stage) or ((solution.slowest_stage==best_solution.slowest_stage) and (solution.edge_cut < best_solution.edge_cut)):
            best_solution = solution

    partition,edge_cut,slowest_stage,volumes,algorithm = best_solution

    for n in graph.nodes:
        n.part = partition[n.id]
    
    cutting_edges=0
    cutting_scopes=[]
    for n in graph.nodes:
        for u in n.out_edges:
            if u.part != n.part:
                cutting_edges +=1
                cutting_scopes.append(n.scope)
                
    print()
    print("-I- Printing Partitioning Report")
    print(f"    objective:{objective.name}")
    if objective is Objective.EDGE_CUT:
        print(f"    objective value: {edge_cut:.2f}")
    else:
        print(f"    objective value: {slowest_stage:.2f}")
    print(f"    best algorithm:{algorithm.name}")
    print(f"    number of cutting edges: {cutting_edges}")
    print(f"    edge cut:{edge_cut:.2f}")
    print(f"    volumes:{volumes}")

    return graph,edge_cut,volumes
        

def worker(kwargs)->Solution:
    kwargs['graph'] = Graph(None, None, None, None,None).load_state(kwargs['graph'])
    seed = kwargs.pop("seed")
    allocated_seconds = kwargs.pop("allocated_seconds")
    objective = kwargs['objective']
    random.seed(seed)
    start = time.time()
    best_solution = Solution(None,np.inf,np.inf,None,None)
    steps=0
    while (time.time() - start) < allocated_seconds:
        solution=_acyclic_partition(**kwargs)
        if objective is Objective.EDGE_CUT:
            if (solution.edge_cut < best_solution.edge_cut) or ((solution.edge_cut==best_solution.edge_cut) and (solution.slowest_stage < best_solution.slowest_stage)):
                best_solution = solution
        elif (solution.slowest_stage < best_solution.slowest_stage) or ((solution.slowest_stage==best_solution.slowest_stage) and (solution.edge_cut < best_solution.edge_cut)):
            best_solution = solution
        steps+=1

    return best_solution


def _acyclic_partition(graph:Graph,algorithm:ALGORITHM=ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES,k:int=4,epsilon:float=0.1,
                    node_weight_function:Optional[NodeWeightFunction]=None,
                    edge_weight_function:Optional[EdgeWeightFunction]=None,
                    objective:Objective=Objective.EDGE_CUT,
                    rounds:int=10,use_layers_graph=True)->Solution:
    if not use_layers_graph:
        work_graph = graph
    else:
        work_graph, layers_to_original = graph.layers_graph()


    if node_weight_function is None:
        node_weight_function = lambda n: 1
    
    if edge_weight_function is None:
        edge_weight_function = lambda u,v: 1
    
    

    node_weights=dict()
    edge_weights=dict()
    
    for n in work_graph.nodes:
        node_weights[n] = node_weight_function(n)
        for o in n.out_edges:
            edge_weights[(n,o)] = edge_weight_function(n,o)

    initial_divide(work_graph,k,node_weights)

    partition_volumes = calculate_partition_volumes(k,node_weights)
    L_max = (1+epsilon)*math.ceil(sum(partition_volumes.values())/k)
    
    msg = "\n".join(["-I- balanced partitioning is not possible",
    f"   max allowed weight: {L_max:.2f}",
    f"   max node weight: {max(node_weights.values()):.2f}"])

    assert all((v <= L_max for v in node_weights.values())),msg  
    
    if objective is Objective.STAGE_TIME:
        partition_volumes=calculate_stage_times(node_weights,edge_weights)

    ALGORITHMS[algorithm](partition_volumes,edge_weights,node_weights,L_max,rounds=rounds,objective=objective)
    
    #refine partition in a greedy fashion
    global_moves(partition_volumes,edge_weights,node_weights,L_max,rounds=1,objective=objective)
    
    # induce partition from the layers graph to the original graph
    # recalculate partition metrics
    if use_layers_graph:
        induce_layer_partition(graph,work_graph, layers_to_original)
        #calculate metrics on original graph
        node_weights=dict()
        edge_weights=dict()
        for n in graph.nodes:
            node_weights[n] = node_weight_function(n)
            for o in n.out_edges:
                edge_weights[(n,o)] = edge_weight_function(n,o)
        if objective is Objective.EDGE_CUT:
            partition_volumes = calculate_partition_volumes(k,node_weights)
        else:
            partition_volumes = calculate_stage_times(node_weights,edge_weights)

    QuotientGraph(graph.nodes).selfcheck()

    edge_cut = calculate_edge_cut(graph.nodes,edge_weights)

    return Solution({n.id:n.part for n in graph.nodes},edge_cut,max(partition_volumes.values()),partition_volumes,algorithm)
