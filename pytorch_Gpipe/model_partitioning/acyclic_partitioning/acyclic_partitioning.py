from pytorch_Gpipe.model_profiling import Graph,Node,NodeWeightFunction,EdgeWeightFunction
from .data_structures import QuotientGraph,PartitionNode,PriorityQueue
from ..METIS_partitioning.partition_graph import induce_layer_partition
import random
import math
import numpy as np
from typing import Tuple,Dict,Optional,Iterator
from collections import defaultdict
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
    #k partitions require k-1 seperators
    while len(Vs) < k-1:
        stage_weight = options[random.randint(0,1)]
        cumulative_node_weights-=stage_weight
        Vs.append(np.searchsorted(cumulative_node_weights,0))

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
def simple_moves(partition_volumes:Dict[int,float],edge_weights:Dict[Tuple[Node,Node],float],node_weights:Dict[Node,float],L_max:float,rounds:int=1):
    #we use 0 based indexing
    k=len(partition_volumes)-1
    for _ in range(rounds):
        changed=False

        # O(E)
        for n in node_weights.keys():
            gain_left = -np.inf
            if (n.part > 0) and (C_in(n,n.part,edge_weights)==0) and ((partition_volumes[n.part-1]+node_weights[n])<L_max):
                gain_left = C_in(n,n.part-1,edge_weights) - C_out(n,n.part,edge_weights)
            
            gain_right = -np.inf
            if (n.part < k) and (C_out(n,n.part,edge_weights)==0) and ((partition_volumes[n.part+1]+node_weights[n])<L_max):
                gain_right = C_out(n,n.part+1,edge_weights) - C_in(n,n.part,edge_weights)

            moves = defaultdict(list)
            moves[gain_left].append(n.part-1)
            moves[gain_right].append(n.part+1)
            
            max_gain = max(moves.keys())
            if max_gain < 0:
                continue

            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves,1)[0]
            
            partition_volumes[n.part] -= node_weights[n]
            partition_volumes[dst] += node_weights[n]
            n.part = dst

        if not changed:
            break


#move nodes between all partitions as long as edge cut improves and constraints are enforced
#uses a sufficient condition for enforcing acyclicicity not a necessary condition
# as such some options are skipped
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def advanced_moves(partition_volumes:Dict[int,float],edge_weights:Dict[Tuple[Node,Node],float],node_weights:Dict[Node,float],L_max:float,rounds:int=1):
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
                gain = calculate_gain(n,j,edge_weights)
                if (partition_volumes[j]+node_weights[n]) > L_max:
                    gain = -np.inf
                moves[gain].append(j)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue
            
            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves,1)[0]
            
            partition_volumes[n.part] -= node_weights[n]
            partition_volumes[dst] += node_weights[n]
            n.part = dst

        if not changed:
            break


#move nodes between all partitions as long as edge cut improves and constraints are enforced
#uses Khan's algorithm to ensure we do not create cycles in the quotient graph
# a move is aligible as long as it does not overload the target partition and does not create a cycle
def global_moves(partition_volumes:Dict[int,float],edge_weights:Dict[Tuple[Node,Node],float],node_weights:Dict[Node,float],L_max:float,rounds:int=1):
    quotient_graph = QuotientGraph(node_weights.keys())

    for _ in range(rounds):
        changed=False

        # O(E(k+mq))
        for n in node_weights.keys():
            moves = defaultdict(list)
            for j in range(len(partition_volumes)):
                if j == n.part:
                    continue

                gain = calculate_gain(n,j,edge_weights)
                if ((partition_volumes[j]+node_weights[n]) > L_max) or quotient_graph.move_creates_cycle(n,j):
                    gain = -np.inf
        
                moves[gain].append(j)

            max_gain = max(moves.keys())
            if max_gain < 0:
                continue
            
            changed = True
            best_moves = moves[max_gain]
            dst = random.sample(best_moves,1)[0]
            partition_volumes[n.part] -= node_weights[n]
            partition_volumes[dst] += node_weights[n]
            
            quotient_graph.move_node(n,dst)
            if DEBUG:
                quotient_graph.selfcheck()

        if not changed:
            break


# move nodes between partitions
# moves with negative gain are also eligible in order to escape local minima
# the partitioning with the best objective will be returned
def Fiduccia_Mattheyses_moves(partition_volumes:Dict[int,float],edge_weights:Dict[Tuple[Node,Node],float],node_weights:Dict[Node,float],L_max:float,rounds:int=1):
    quotient_graph = QuotientGraph(node_weights.keys())

    best_objective = calculate_edge_cut(node_weights.keys(),edge_weights)

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
                    candidate_moves.push_task(calculate_gain(node,B,edge_weights),(node,B))
            for node in quotient_graph[B]:
                if all(i.part <= A for i in node.in_edges):
                    candidate_moves.push_task(calculate_gain(node,A,edge_weights),(node,A))
            
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

                gain = calculate_gain(node,dst,edge_weights)
                partition_volumes[node.part] -= node_weights[node]
                partition_volumes[dst] += node_weights[node]
                current_objective -= gain
                
                src = node.part
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
                            gain = calculate_gain(i,B,edge_weights)
                            candidate_moves.push_task(gain,(i,B))
                else:
                    for o in node.out_edges:
                        if o.part == B and all(i.part <= A for i in o.in_edges):
                            gain = calculate_gain(o,A,edge_weights)
                            candidate_moves.push_task(gain,(o,A))

            #end of inner pass revert partition to best partition
            for n,dst in moves_to_best.items():
                partition_volumes[n.part]-=node_weights[n]
                partition_volumes[dst]+=node_weights[n]
                quotient_graph.move_node(n,dst)
            if DEBUG:
                quotient_graph.selfcheck()


#assumes W(u,v) > 0

def calculate_gain(v:Node,dest:int,edge_weights:Dict[Tuple[Node,Node],float])->float:
    return C_in(v,dest,edge_weights) - C_out(v,v.part,edge_weights) + C_out(v,dest,edge_weights) - C_in(v,v.part,edge_weights)


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


ALGORITHMS = {
    ALGORITHM.SIMPLE_MOVES:simple_moves,
    ALGORITHM.ADVANCED_MOVES:advanced_moves,
    ALGORITHM.GLOBAL_MOVES:global_moves,
    ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES:Fiduccia_Mattheyses_moves
}

def acyclic_partition(graph:Graph,k:int,epsilon:float=0.1,node_weight_function:Optional[NodeWeightFunction]=None,
                    edge_weight_function:Optional[EdgeWeightFunction]=None,rounds:int=10,allocated_seconds:int=10,use_layers_graph:bool=True)->Tuple[Graph,float,Dict[int,float]]:
    worker_args=[dict(graph = graph.state(),
                        k=k,
                        algorithm=alg,
                        epsilon=epsilon,
                        node_weight_function=node_weight_function,
                        edge_weight_function=edge_weight_function,
                        rounds=rounds,
                        allocated_seconds=allocated_seconds,
                        seed=random.randint(0,2**32),
                        use_layers_graph=use_layers_graph) for alg in ALGORITHM]

    with Pool(len(worker_args)) as pool:
        results=pool.map(worker, worker_args)

    assert len(results) == len(ALGORITHMS)

    best_partition = results[0]
    for r in results:
        if r[1] < best_partition[1]:
            best_partition = r
        elif (r[1] == best_partition[1]) and (max(r[2].values()) < max(best_partition[2].values())):
            best_partition = r

    partition,edge_cut,volumes,algorithm = best_partition

    for n in graph.nodes:
        n.part = partition[n.id]
    
    regular_edges=0
    penalty_edges=0
    penalty_scopes=[]
    for n in graph.nodes:
        for u in n.out_edges:
            if u.part != n.part:
                if edge_weight_function(n,u) >= 1000:
                    penalty_edges+=1
                    penalty_scopes.append(n.scope)
                else:
                    regular_edges+=1
    print()
    print("-I- Printing Partitioning Report")
    print(f"    best algorithm:{algorithm.name}")
    print(f"    number of cutting edges: {regular_edges+penalty_edges}")
    print(f"    number of regular edges: {regular_edges}")
    print(f"    number of penalty edges: {penalty_edges}")
    print(f"    edge cut:{edge_cut:.2f}")
    print(f"    volumes:{volumes}")

    return graph,edge_cut,volumes
        

def worker(kwargs)->Tuple[Dict[int,int],float,Dict[int,float],ALGORITHM]:
    kwargs['graph'] = Graph(None, None, None, None,None).load_state(kwargs['graph'])
    seed = kwargs.pop("seed")
    allocated_seconds = kwargs.pop("allocated_seconds")
    random.seed(seed)
    start = time.time()
    bp,be,bv = None,np.inf,None
    steps=0
    while (time.time() - start) < allocated_seconds:
        p,e,v=_acyclic_partition(**kwargs)
        if e < be:
            bp,be,bv = p,e,v
        elif (e == be) and (max(v.values()) < max(bv.values())):
            # same edge cut take the one with fastest worst case
            bp,be,bv = p,e,v
        steps+=1
    return bp,be,bv,kwargs['algorithm']


def _acyclic_partition(graph:Graph,algorithm:ALGORITHM=ALGORITHM.FIDUCCIA_MATTHEYSES_MOVES,k:int=4,epsilon:float=0.1,
                    node_weight_function:Optional[NodeWeightFunction]=None,
                    edge_weight_function:Optional[EdgeWeightFunction]=None,
                    rounds:int=10,use_layers_graph=True)->Tuple[Dict[int,int],float,Dict[int,float]]:
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
    ALGORITHMS[algorithm](partition_volumes,edge_weights,node_weights,L_max,rounds=rounds)
    
    #refine partition in a greedy fashion
    global_moves(partition_volumes,edge_weights,node_weights,L_max,rounds=1)
    
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
        partition_volumes = calculate_partition_volumes(k,node_weights)

    QuotientGraph(graph.nodes).selfcheck()

    edge_cut = calculate_edge_cut(graph.nodes,edge_weights)

    return {n.id:n.part for n in graph.nodes},edge_cut,partition_volumes
