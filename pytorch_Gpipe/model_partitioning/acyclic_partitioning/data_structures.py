from collections import defaultdict
from pytorch_Gpipe.model_profiling import Node
from typing import Dict,Iterable,Set,Iterator,List,Tuple,Any
import random
import heapq


class PriorityQueue():
    #heapq is a min heap and we need a max heap
    #so we push the negative gain
    #we use a random tie braker in case several tasks have the same gain
    def __init__(self):
        self.heap=[]
    
    def push_task(self,gain:float,task:Any):
        tie_braker = random.randint(0,2**32)
        priority = (-gain,-tie_braker)

        heapq.heappush(self.heap,(priority,task))
    
    def pop_task(self)->Any:
        priority,task = heapq.heappop(self.heap)
        return task

    def __len__(self)->int:
        return len(self.heap)
    
    def __bool__(self)->bool:
        return len(self) > 0


class PartitionNode():
    """ PartitionNode is a collection of graph nodes allocated to the same partition
        an edge exists between PartitionNodes iff they there are edges between the underlying graph nodes
    """
    def __init__(self,nodes:Iterable[Node],idx:int):
        self.nodes:Set[Node] = set(nodes)
        self._out_edges = defaultdict(lambda : 0)
        self._in_edges = defaultdict(lambda : 0)
        self.idx=idx

        for n in self.nodes:
            for i in n.in_edges:
                self._in_edges[i.part]+=1
            for o in n.out_edges:
                self._out_edges[o.part]+=1
        
        self._out_edges.pop(self.idx,None)
        self._in_edges.pop(self.idx,None)
    
    @property
    def in_edges(self)->List[int]:
        return [i for i,n in self._in_edges.items() if n > 0]
    
    @property
    def out_edges(self)->List[int]:
        return [i for i,n in self._out_edges.items() if n > 0]
    
    def __contains__(self, key)->bool:
        return key in self.nodes

    def __iter__(self)->Iterator[Node]:
        return iter(self.nodes)

    def __len__(self)->int:
        return len(self.nodes)

    def add_in_edge(self,src:int):
        self._in_edges[src]+=1
    
    def add_out_edge(self,dst:int):
        self._out_edges[dst]+=1

    def remove_in_edge(self,src:int):
        self._in_edges[src]-=1
    
    def remove_out_edge(self,dst:int):
        self._out_edges[dst]-=1

    def add_node(self,node:Node):
        self.nodes.add(node)
    
    def remove_node(self,node:Node):
        self.nodes.discard(node)


class QuotientGraph():
    DEBUG=False
    def __init__(self,nodes:Iterable[Node]):
        groups = defaultdict(list)
        for n in nodes:
            groups[n.part].append(n)
        
        self._nodes:Dict[int,PartitionNode] = {idx:PartitionNode(group,idx) for idx,group in groups.items()}
        
        if self.DEBUG:
            assert not self.has_cycles()
            self.selfcheck()
    
    def __getitem__(self,idx:int)->PartitionNode:
        return self._nodes[idx]

    def move_node(self,node:Node,dst:int):
        assert node.part != dst
        src=node.part
        src_part =self[src]
        dst_part =self[dst]
        
        src_part.remove_node(node)
        dst_part.add_node(node)
        node.part = dst

        for i in node.in_edges:
            i_part = self[i.part]
            #remove edge from i to src
            src_part.remove_in_edge(i.part)
            i_part.remove_out_edge(src)

            #add edge from i to dest
            i_part.add_out_edge(dst)
            dst_part.add_in_edge(i.part)
        
        for o in node.out_edges:
            o_part = self[o.part]
            #remove edge from src to o
            src_part.remove_out_edge(o.part)
            o_part.remove_in_edge(src)

            #add edge from dst to o
            o_part.add_in_edge(dst)
            self[dst].add_out_edge(o.part)

        #remove self edges
        #faster than using if statements in the for loops
        for p in self.nodes:
            p._in_edges.pop(p.idx,None)
            p._out_edges.pop(p.idx,None)

        if self.DEBUG:
            self.selfcheck()
    
    def move_creates_cycle(self,node:Node,dest:int)->bool:
        orig_part=node.part
        self.move_node(node,dest)
        creates_cycle = self.has_cycles()
        self.move_node(node,orig_part)
        return creates_cycle

    @property
    def nodes(self)->Iterable[PartitionNode]:
        return self._nodes.values()

    def has_cycles(self)->bool:
        S=[]
        T=[]
        degs=dict()
        # o(V)
        for n in self.nodes:
            assert isinstance(n,PartitionNode)
            if len(n.in_edges)==0:
                S.append(n.idx)
            else:
                degs[n]=len(n.in_edges)

        # O(E)
        while S:
            n = self._nodes[S.pop()]
            assert isinstance(n,PartitionNode)
            T.append(n)
            for o in n.out_edges:
                out = self._nodes[o]
                assert isinstance(out,PartitionNode)
                degs[out]-=1
                if degs[out] == 0:
                    S.append(o)


        #if we have a cycle S will be empty and T will not contain all of the nodes
        return len(T) < len(self.nodes)

    def build_dot(self):
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

        colors = {
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

        # add nodes
        for node in self.nodes:
            dot.node(str(node.idx), label=f"partition:{node.idx}",
                        fillcolor=colors[node.idx])
            for i in node.in_edges:
                dot.edge(str(i), str(node.idx))

        return dot

    def save_as_pdf(self,file_name: str,directory: str):
        '''
        save the rendered graph to a pdf file

        Parameters
        ----------
        file_name:
            the name of the saved file
        directory:
            directory to store the file in
        '''
        dot = self.build_dot()
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)
        return self
    
    def print_stats(self,node_weights:Dict[Node,float],edge_weights:Dict[Tuple[int,int],float]):
        volumes = defaultdict(lambda: 0)
        edge_cut=0
        number_of_cutting_edges=0
        for partition in self.nodes:
            for n in partition:
                volumes[partition.idx]+=node_weights[n]
                for o in n.out_edges:
                    if n.part != o.part:
                        if edge_weights[(n.id,o.id)] >= 1000:
                            print(f"{n.id}=>{o.id}")
                            print(f"{n.part}=>{o.part}")
                            print(f"{n.value_type}")
                            print(f"weight:{edge_weights[(n.id,o.id)]:.2f}\n")
                        edge_cut+=edge_weights[(n.id,o.id)]
                        number_of_cutting_edges+=1

        total_volume = sum(volumes.values())
        avg_volume = total_volume/len(volumes)
        print(f"total number of nodes: {len(node_weights)}")
        print(f"total number of edges: {len(edge_weights)}")
        print(f"total weight: {total_volume:.2f}")
        print(f"avg weight: {avg_volume:.2f}")
        print(f"number of cutting edges: {number_of_cutting_edges}")
        print(f"edge cut: {edge_cut:.2f}")
        print("partition stats:")
        for i in range(len(volumes)):
            print(f"    partition {i}")
            print(f"    number of nodes {len(self._nodes[i])}")
            print(f"    partition volume: {volumes[i]:.2f}\n")

    def selfcheck(self):
        visited = set()
        for idx,n in self._nodes.items():
            assert idx == n.idx
            for u in n.nodes:
                assert u.part == idx
                assert u not in visited
                visited.add(u)

            for i,v in n._in_edges.items():
                assert v >=0, (idx,i,v)
            assert idx not in n._in_edges
            for i in n.in_edges:
                assert idx in self._nodes[i].out_edges,(idx,i)
            
            for o,v in n._out_edges.items():
                assert v >=0, (idx,o,v)
            assert idx not in n._out_edges
            for o in n.out_edges:
                assert idx in self._nodes[o].in_edges,(idx,o)

        assert not self.has_cycles()