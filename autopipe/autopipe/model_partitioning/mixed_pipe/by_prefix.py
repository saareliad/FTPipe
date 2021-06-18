import warnings
from collections import defaultdict
from pprint import pprint

from torch.nn import Module

from autopipe.autopipe.model_partitioning.mixed_pipe.check_cycles import check_cycle2
from autopipe.autopipe.model_profiling.control_flow_graph import Graph
from autopipe.autopipe.union_find import UnionFind
from autopipe.autopipe.utils import special_traverse_model


def coarsen_prefixes(model: Module, graph: Graph, node_weight_function, edge_weight_function, uf: UnionFind,
                     basic_blocks, special_blocks, depth):
    prev_graph = Graph.from_other(graph)
    # Used to find the local multi-matching
    uf2 = UnionFind(elements=graph._nodes.keys())

    # FIXME: replace by finer granularity:
    nodes = list(graph.non_input_nodes)
    sb_scope_to_nodes = get_marked_nodes_for_prefix_coarsening(module=model,
                                                            nodes=nodes,
                                                            basic_blocks=basic_blocks,
                                                            special_blocks=special_blocks,
                                                            depth=depth)
    # will only merge nodes under the same sb scope.
    for sb_scope, sb_nodes in sb_scope_to_nodes.items():
        set_sb_nodes = set(sb_nodes)
        sb_nodes.sort(key=lambda n: n.topo_sort_id)

        did_something = True
        while did_something and len(set_sb_nodes) > 1:
            did_something = False
            for u in sb_nodes:
                if u not in set_sb_nodes:
                    continue

                for v in sorted(u.out_edges, key=lambda n: n.topo_sort_id):
                    if v not in set_sb_nodes:
                        continue

                    if check_cycle2(graph, u, v):
                        # can't merge without breaking topo sort
                        continue
                    graph.merge(uid=u.id, vid=v.id, edge_weight_function=edge_weight_function, uf=uf)
                    # graph.topo_sort(change_graph=False)

                    uf.union(u.id, v.id)
                    uf2.union(u.id, v.id)
                    set_sb_nodes.discard(v)
                    did_something = True

        if len(set_sb_nodes) > 1:
            warnings.warn(f"failed to fully coarsen special block. remaining ({len(set_sb_nodes)}): {set_sb_nodes}")
            raise NotImplementedError()

    matching = None
    sb_names = list(sb_scope_to_nodes.keys())
    # print("Found special block names:", sb_names)
    return prev_graph, matching, graph, uf, uf2, sb_names


def get_marked_nodes_for_prefix_coarsening(module, nodes, basic_blocks, special_blocks, depth):
    sb_id_to_nodes = dict()

    all_sbs = []
    for packed in special_traverse_model(module, depth=depth,
                                         basic_blocks=basic_blocks,
                                         special_blocks=special_blocks,
                                         full=True,
                                         mark=True):
        sub_layer, scope, parent, terminal, sb_id = packed
        if sb_id is not None:
            all_sbs.append(packed)

    # pprint(list((a[1], a[4]) for a in all_sbs))

    for packed in all_sbs:
        _, scope, _, _, sb_id = packed
        l= []
        l = [node for node in nodes if (node.scope.startswith(scope) or (node.scope_to_hold_to and node.scope_to_hold_to.startswith(scope)))]
        sb_id_to_nodes[scope] = l
    return sb_id_to_nodes


def annotate_special_blocks_to_hold_to(model, graph, special_blocks, basic_blocks, depth):

    # step1: find the core basic blocks, to extract their scopes.
    nodes = list(graph.non_input_nodes)
    sb_scope_to_nodes = get_marked_nodes_for_prefix_coarsening(module=model,
                                                               nodes=nodes,
                                                               basic_blocks=basic_blocks,
                                                               special_blocks=special_blocks,
                                                               depth=depth)

    scopes_to_hold_to = list(sb_scope_to_nodes.keys())
    # step2: find nodes which have this scopes as prefix i.e., children of each SB node.
    # step 3: annotate children.
    for node in graph.nodes:
        for scope_to_hold_to in scopes_to_hold_to:
            if node.scope.startswith(scope_to_hold_to):
                if node.scope_to_hold_to is not None and node.scope_to_hold_to != scope_to_hold_to:
                    print(f"need to assign a scope to hold to for node:{node.scope}")
                    print(f"but node already has {node.scope_to_hold_to}")
                    raise NotImplementedError("nested by prefix coarsening not supported")
                node.scope_to_hold_to = scope_to_hold_to

