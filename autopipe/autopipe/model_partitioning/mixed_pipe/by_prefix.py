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

    nodes = list(graph.non_input_nodes)
    sb_id_to_nodes = mark_nodes_for_prefix_coarsening(module=model,
                                                      nodes=nodes,
                                                      basic_blocks=basic_blocks,
                                                      special_blocks=special_blocks,
                                                      depth=depth)
    for sb_scope, sb_nodes in sb_id_to_nodes.items():
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
    sb_names = list(sb_id_to_nodes.keys())
    print("Found special block names:", sb_names)
    return prev_graph, matching, graph, uf, uf2, sb_names


def mark_nodes_for_prefix_coarsening(module, nodes, basic_blocks, special_blocks, depth):
    first_sb_id_to_scopes = dict()
    sb_id_to_nodes = dict()
    sb_class_to_first_sb_ids = defaultdict(list)

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
        l = [node for node in nodes if node.scope.startswith(scope)]
        sb_id_to_nodes[scope] = l
    return sb_id_to_nodes
