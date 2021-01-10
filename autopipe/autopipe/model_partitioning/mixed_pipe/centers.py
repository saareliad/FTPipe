import math
import random
from collections import deque
from pprint import pprint

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node
from autopipe.autopipe.union_find import UnionFind


def stochastic_centers_matching(graph: Graph, node_weight_function: NodeWeightFunction,
                                edge_weight_function: EdgeWeightFunction,
                                L, P, uf: UnionFind,
                                verbose=False, record_history=False):
    print("stochastic_centers_matching")
    prev_graph = Graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())

    all_nodes = {n for n in graph.non_input_nodes}

    # choose L_centers()
    # get basic blocks names
    bb = graph.basic_blocks
    bb_names = [c.__name__ for c in bb]
    # find nodes which are basic blocks.

    found_nodes = {b: list() for b in bb_names}
    total_found = 0

    for n in graph.non_input_nodes:
        for b in bb_names:
            if b in n.scope:
                found_nodes[b].append(n)
                total_found += 1

    print(f"Found {total_found} basic blocks")
    pprint(found_nodes)

    if total_found < L:
        raise NotImplementedError("random")

    print("assigning centers from basic blocks")
    lengths = {b: math.floor(L * (len(nodes) / total_found)) for b, nodes in found_nodes.items()}
    total_basic_block_centers = sum(lengths.values())
    print(f"total_basic_block_centers: {total_basic_block_centers}")
    print("centers per basic block:")
    pprint(lengths)

    # split L to each bb size

    # TODO maybe some loop here, since the rest should be random.
    # centers
    hd = deque()
    centers = set()
    to_assign = L
    sorted_iter = sorted(list(found_nodes.items()), key=lambda x: len(x[1]))
    for b_name, nodes in sorted_iter:

        L_tag = len(nodes)
        L_prop_int = lengths[b_name]
        jump = math.ceil(L_tag / L_prop_int)  # FIXME: see above
        if jump <= 0:
            continue
        for i in range(0, L_tag, jump):
            center = nodes[i]
            hd.append(center)
            centers.add(center)
            to_assign -= 1
            if to_assign == 0:
                break

        if to_assign == 0:
            break


    if to_assign > 0:
        # raise NotImplementedError("need to randomize")
        print(f"choosing {to_assign} random centers")
        # TODO: choose nodes with maximal distance from graph input and last node.
        additional_centers = random.sample(all_nodes - centers, to_assign)

        for x in additional_centers:
            centers.add(x)
            hd.append(x)
        to_assign -= len(additional_centers)
        assert to_assign == 0

    print("centers")
    print(hd)

    def inner_loop():
        # round robin on centers, attempt to merge u<----v
        for i in range(len(hd)):
            # Try to find match:
            u = hd.popleft()
            for v in sorted(u.out_edges, key=lambda n: n.topo_sort_id):
                if v in centers:
                    continue  # TODO can check and pop.
                if check_cycle2(graph, u, v):
                    # can't merge without breaking topo sort
                    continue
                graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
                uf.union(u.id, v.id)
                uf2.union(u.id, v.id)
                all_nodes.discard(v)
                hd.append(u)
                return True
            hd.append(u)
        return False

    history_sizes = []
    history_weights = []
    while len(all_nodes) > L:
        merged_something = inner_loop()
        if not merged_something:
            break
        if record_history:
            history_sizes.append(len(all_nodes) + 1)
            # history_weights.append(weight_of_u)
        if verbose:
            print(f"Nodes: {len(all_nodes)} Centers: {len(hd)}")



    if len(all_nodes)  > L:
        print(f"Merged until {len(all_nodes)} Merging more, until {L} left")

        def inner_loop():
            # round robin on centers, attempt to merge u<----v
            for i in range(len(hd)):
                # Try to find match:
                v = hd.popleft()
                v: Node
                for u in sorted(v.in_edges, key=lambda n: -n.topo_sort_id):
                    if u not in all_nodes:
                        continue
                    if u in centers:
                        continue  # TODO can check and pop.
                    if check_cycle2(graph, u, v):
                        # can't merge without breaking topo sort
                        continue
                    graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
                    uf.union(u.id, v.id)
                    uf2.union(u.id, v.id)
                    all_nodes.discard(v)
                    centers.discard(v)
                    centers.add(u)
                    hd.append(u)
                    return True
                hd.append(v)
            return False


        while len(all_nodes) > L:
            merged_something = inner_loop()
            if not merged_something:
                break
            if record_history:
                history_sizes.append(len(all_nodes) + 1)
                # history_weights.append(weight_of_u)
            if verbose:
                print(f"Nodes: {len(all_nodes)} Centers: {len(hd)}")

    # Note: matching is pretty much meaningless.
    matching = None
    return prev_graph, matching, graph, uf, uf2
