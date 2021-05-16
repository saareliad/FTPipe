import math
import random
import warnings
from collections import deque
from pprint import pprint

from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction, EdgeWeightFunction
from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_profiling.control_flow_graph import Graph, Node
from autopipe.autopipe.union_find import UnionFind


def stochastic_centers_matching(graph: Graph, node_weight_function: NodeWeightFunction,
                                edge_weight_function: EdgeWeightFunction,
                                L, P, uf: UnionFind,
                                verbose=False, record_history=False,
                                special_blocks=None,
                                sb_names=None):
    print("stochastic_centers_matching")
    prev_graph = Graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())
    all_nodes = {n for n in graph.non_input_nodes}

    # choose L_centers()
    # get basic blocks names
    if special_blocks is None:
        special_blocks = ()

    bb = special_blocks
    sb_names = [c.__name__ for c in bb]
    found_nodes = {b: list() for b in sb_names}
    total_found = 0

    # find nodes which are basic blocks.
    for n in graph.non_input_nodes:
        for b in sb_names:
            if b in n.scope or (n.scope_to_hold_to and b in n.scope_to_hold_to):
                found_nodes[b].append(n)
                total_found += 1

    print(f"-I- Found {total_found} special blocks")
    pprint(found_nodes)

    # if sb_names is not None:
    # TODO: I'm fixing the merging of special blocks.
    # need to make names compact e.g 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[15]'
    # we need to use "special blocks cmd_arg for the matther as I use basic blocks from the graph.

    # to_assign = L
    if total_found < L:
        warnings.warn(f"There are only {total_found} special blocks, but need to find {L} centers")
        warnings.warn("Finding {L-total_found} more random centers, all found special block centers will be centers")
        # to_assign = total_found
        # needtofind = L-total_found
        # raise NotImplementedError("random centers")

    print("-I- assigning centers from special blocks")
    lengths = {b: math.floor(L * (len(nodes) / total_found)) for b, nodes in found_nodes.items()}
    total_basic_block_centers = sum(lengths.values())  # TODO: rename to special blocks
    print(f"-I- total_basic_block_centers: {total_basic_block_centers}")
    print(f"-I- centers to assign in each basic block: {lengths}")
    # pprint(lengths)

    # split L to each bb size

    # TODO maybe some loop here, since the rest should be random.
    # centers
    hd = deque()
    centers = set()
    to_assign = L
    sorted_iter = sorted(list(found_nodes.items()), key=lambda x: len(x[1]))
    for b_name, nodes in sorted_iter:
        print(f"-I- Assigning centers in block {b_name}")
        L_tag = len(nodes)
        L_prop_int = lengths[b_name]   # The proportion of centers this block gets
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
    print(f"-I- Assigned total of {len(centers)} centers:")
    pprint(centers)

    if to_assign > 0:
        print(f"-I- Now, choosing {to_assign} more random centers")
        # TODO: choose nodes with maximal distance from graph input and last node.
        additional_centers = random.sample(all_nodes - centers, to_assign)

        for x in additional_centers:
            centers.add(x)
            hd.append(x)
        to_assign -= len(additional_centers)
        assert to_assign == 0

    print("-I- final centers:")
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
