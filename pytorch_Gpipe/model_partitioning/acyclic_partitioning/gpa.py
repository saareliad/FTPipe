import random
from typing import Dict,Tuple,List,Optional
from pytorch_Gpipe.model_profiling import Node,Graph
from pytorch_Gpipe.model_partitioning.acyclic_partitioning.data_structures import Path,PathSet,ContractedGraph,SimpleNode

#adapted from https://github.com/KaHIP/KaHIP/blob/master/lib/partition/coarsening/matching/gpa/gpa_matching.cpp


def coarsening(graph:Graph,node_weights:Dict[SimpleNode,float],
                    edge_weights:Dict[Tuple[SimpleNode,SimpleNode],float])->List[Tuple[ContractedGraph,Dict[int,int],ContractedGraph]]:
    g = ContractedGraph.from_Graph(graph,node_weights,edge_weights)
    n = len(g)
    
    hierarchy=[]
    p = g
    while True:
        matching,_ = find_max_matching(g._node_weights,g._edge_weights)
        g=g.contract(g,matching)
        if len(g) == n:
            break
        hierarchy.append((p,matching,g))
        n = len(g)
        p = g
    
    return hierarchy


def refine(fine_graph:ContractedGraph,coarse_graph:ContractedGraph,matching:Dict[int,int]):
    for n in fine_graph.nodes:
        n.stage_id = coarse_graph[matching[n.id]].stage_id


def find_max_matching(node_weights:Dict[SimpleNode,float],
                    edge_weights:Dict[Tuple[SimpleNode,SimpleNode],float])->Tuple[Dict[int,int],float]:
    edges = list(edge_weights.keys())
    edge_ratings = {e: edge_rating(e[0],e[1],edge_weights,node_weights) for e in edges}
    random.shuffle(edges)
    edges = sorted(edges,key=lambda e: edge_ratings[e],reverse=True)
    
    pathset = PathSet(node_weights.keys())
    #find all paths and cycles
    for edge in edges:
        pathset.add_if_eligible(edge)


    max_match = []
    max_match_weight = 0
    #find maximum matching for each path and cycle
    for node in node_weights.keys():
        path = pathset.paths[node]
        if not path.active:
            continue
        if path.end is not node:
            continue
        if path.length == 0:
            continue

        if path.is_cycle():
            unpacked_cycle = unpack_path(pathset,path)
            first_edge = unpacked_cycle.pop(0)
            match_a,match_a_weight = max_path_matching(unpacked_cycle,edge_ratings)
            unpacked_cycle.insert(0,first_edge)

            last_edge = unpacked_cycle.pop()
            match_b,match_b_weight = max_path_matching(unpacked_cycle,edge_ratings)
            unpacked_cycle.append(last_edge)

            if match_a_weight > match_b_weight:
                match = match_a
                match_weight = match_a_weight
            else:
                match = match_b
                match_weight = match_b_weight
                
        elif path.length == 1:
            if pathset.next_vertex(path.end) is path.start:
                edge = pathset.edge_to_next(path.end)
            else:
                edge = pathset.edge_to_prev(path.end)
                assert pathset.next_vertex(path.end) is path.start
            match,match_weight = [edge],edge_ratings[edge]
        else:
            unpacked_path = unpack_path(pathset,path)
            match,match_weight = max_path_matching(unpacked_path,edge_ratings)
        
        max_match.extend(match)
        max_match_weight+=match_weight

    matching = {u.id:v.id for u,v in max_match}
    for n in node_weights.keys():
        if n.id not in matching:
            matching[n.id]=n.id

    return matching,max_match_weight


def max_path_matching(unpacked_path:List[Tuple[Node,Node]],edge_ratings:Dict[Tuple[Node,Node],float])->Tuple[List[Tuple[Node,Node]],float]:
    k = len(unpacked_path)
    if k == 1:
        #make sure to make a copy
        return list(unpacked_path),edge_ratings[unpacked_path[0]]
    
    ratings = [0]*k
    decision = [False]*k


    ratings[0] = edge_ratings[unpacked_path[0]]
    ratings[1] = edge_ratings[unpacked_path[1]]

    decision[0] = True
    if ratings[0] < ratings[1]:
        decision[1] = True
    
    #find optimal solution via dynamic programing
    for i in range(2,k):
        cur_w = edge_ratings[unpacked_path[i]]
        if(cur_w + ratings[i-2]) > ratings[i-1]:
            decision[i] = True
            ratings[i] = cur_w + ratings[i-2]
        else:
            decision[i] = False
            ratings[i] = ratings[i-1]

    if decision[-1]:
        match_weight = ratings[-1]
    else:
        match_weight = ratings[-2]
    
    match=[]
    i = k-1
    while i >= 0:
        if decision[i]:
            match.append(unpacked_path[i])
            i-=2
        else:
            i-=1
    
    return match,match_weight


def unpack_path(pathset:PathSet,path:Path)->List[Tuple[Node,Node]]:
    assert path.active

    head = path.start
    prev = path.end
    next_v = None
    current = prev

    unpacked_path = []

    if prev is head:
        #cycle
        current = pathset.next_vertex(prev)
        unpacked_path.append(pathset.edge_to_next(prev))

    while current is not head:
        if pathset.next_vertex(current) is prev:
            next_v = pathset.prev_vertex(current)
            unpacked_path.append(pathset.edge_to_prev(current))
        else:
            next_v = pathset.next_vertex(current)
            unpacked_path.append(pathset.edge_to_next(current))

        prev,current = current,next_v
    
    return unpacked_path


def edge_rating(u:Node,v:Node,edge_weights:Dict[Tuple[Node,Node],float],node_weights:Dict[Node,float])->float:
    return (edge_weights[(u,v)]**2)/(1 + node_weights[u] * node_weights[v])


def visualize_matching(nodes,matching,file_name: str,directory: str):
    '''
    return a graphviz representation of the graph
    Parameters
    ----------
    '''
    theme = {
        "background_color": "#FFFFFF",
        "fill_color": "#E8E8E8",
        "outline_color": "#000000",
        "font_color": "#000000",
        "font_name": "Times",
        "font_size": "10",
        "margin": "0,0",
        "padding": "1.0,0.5"
    }
    from graphviz import Digraph

    dot = Digraph()
    dot.attr("graph",
                concentrate="true",
                bgcolor=theme["background_color"],
                color=theme["outline_color"],
                fontsize=theme["font_size"],
                fontcolor=theme["font_color"],
                fontname=theme["font_name"],
                margin=theme["margin"],
                rankdir="TB",
                pad=theme["padding"])

    dot.attr("node",
                shape="box",
                style="filled",
                margin="0,0",
                fillcolor=theme["fill_color"],
                color=theme["outline_color"],
                fontsize=theme["font_size"],
                fontcolor=theme["font_color"],
                fontname=theme["font_name"])

    dot.attr("edge",
                style="solid",
                color=theme["outline_color"],
                fontsize=theme["font_size"],
                fontcolor=theme["font_color"],
                fontname=theme["font_name"])

    partition_color = {
        0: 'grey',
        1: 'green',
        2: 'red',
        3: 'yellow',
        4: 'orange',
        5: 'brown',
        6: 'purple',
        7: 'pink',
        8: 'cyan',
        9: 'gold',
        10: 'darkolivegreen',
        11: 'seagreen',
        12: 'thistle',
        13: 'plum',
        14: 'deeppink',
        15: 'lightyellow',
        16: 'tan'
    }

    def random_color():
        return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    matching_colors = set()
    while len(matching_colors) < len(matching):
        c = random_color()
        if c not in partition_color:
            matching_colors.add(c)

    edge_colors={(u,v):c for (u,v),c in zip(matching.items(),matching_colors)}

    # add nodes
    for node in nodes:
        dot.node(str(node.id), label=node.scope,
                    fillcolor=partition_color[node.stage_id])
        for i in node.in_edges:
            dot.edge(str(i.id), str(node.id),color=edge_colors.get((i.id,node.id),"#000000"))

    dot.format = "pdf"
    import os
    if os.path.exists(f"{directory}/{file_name}.pdf"):
        os.remove(f"{directory}/{file_name}.pdf")
    dot.render(file_name, directory=directory, cleanup=True)