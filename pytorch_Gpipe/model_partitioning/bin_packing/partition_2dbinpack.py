import warnings
from collections import deque, defaultdict
from enum import IntEnum
from itertools import count
from pprint import pprint
from typing import Optional, List, Dict, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pytorch_Gpipe.model_partitioning.bin_packing.heap_dict import heapdict
from pytorch_Gpipe.model_partitioning.bin_packing.post_process import post_process_partition
from pytorch_Gpipe.model_partitioning.bin_packing.union_find import UnionFind
from pytorch_Gpipe.model_partitioning.heuristics import NodeWeightFunction
from pytorch_Gpipe.model_profiling import Graph, Node


# from ...model_profiling import Graph, Node, NodeWeightFunction


# from .partition_2dbinpack_poc import analyze_n_clusters, first_fit_cluster, make_clusters
# Poc abstraction for 2d-packing for pipeline with virtual stages
# ######## ######### ############ ############# ########## ########### ####### ############
# sorts nodes by value
# cluster into similar sizes with n_1, n_2, ..., n_C sizes
# for each cluster i=1...C:
# sort by sequential (id)
# take N =  n_i // K values, and maketree.
# ######## ######## ############# ############## ######### ########### ######## ###########

class ReminderPolicy(IntEnum):
    ToLast = 0
    ToMin = 1


class SecondAndOnClusterPolicy(IntEnum):
    FirstFitBinPacking = 0
    InOrder = 1
    Reversed = 2


def maketree(n, iterable):
    d = deque(iterable)
    res = []
    while d:
        pair = [d.popleft() for _ in range(n)]
        res.append(pair)
    return res


def flatten_subsplit(subsplit):
    to_add = []
    for i in subsplit:
        if isinstance(i, list):
            to_add.extend(i)
        else:
            to_add.append(i)
    return to_add


def sum_subsplit_weight(subsplit):
    return sum(i.weight for i in flatten_subsplit(subsplit))


def get_all_splits(K: int, clusters, id_to_node: Dict[int, Node], to_unify: Dict[int, List[Union[List, Any]]], C: int,
                   reminder_policy: ReminderPolicy = ReminderPolicy.ToLast):
    all_splits = []

    # Make splits
    # Change data type: Pandas object. (id->Index)
    clusters = [list(clusters.groupby("cluster").get_group(c).sort_values("id").itertuples()) for c in range(C)]

    if len(clusters) > 2:
        raise NotImplementedError()

    # clusters: List[List[Any]]
    clusters_lengths = {i: len(clusters[i]) for i in range(len(clusters))}
    print("cluster_lengths, before compressing", clusters_lengths)

    # nest clusters with zero
    new_clusters = []
    for c_i, cluster in enumerate(clusters):
        to_unify_for_cluster = to_unify[c_i]
        cluster_D = {i.Index: i for i in cluster}
        for l in to_unify_for_cluster:
            z = l[0]
            x = l[1]
            if isinstance(cluster_D[x], list):
                v = cluster_D[x]
                zz = cluster_D[z]
                if isinstance(zz, list):
                    v.extend(zz)
                else:
                    v.append(zz)
                v.sort(key=lambda y: y.Index)
            else:
                v = [cluster_D[x]]
                zz = cluster_D[z]
                if isinstance(zz, list):
                    v.extend(zz)
                else:
                    v.append(zz)
                v.sort(key=lambda y: y.Index)
                cluster_D[x] = v

            del cluster_D[z]

        cluster = []
        for i in sorted(cluster_D.keys()):
            cluster.append(cluster_D[i])
        new_clusters.append(cluster)
    clusters = new_clusters

    clusters_lengths = {i: len(clusters[i]) for i in range(len(clusters))}
    print("cluster_lengths, after compressing", clusters_lengths)

    for c_i, cluster in enumerate(clusters):
        n_i = len(cluster)
        reminder = n_i % K

        if n_i < K:
            raise NotImplementedError(f"insufficient number of items in cluster {c_i}, {n_i}, {K}")
        if reminder > 0:
            warnings.warn(
                f"cluster {c_i} is problematic {c_i}, {n_i}%{K}!=0, will put reminding {reminder} nodes in last partition")
            # raise NotImplementedError(f"{c_i}, {n_i}, {K}")
        N = n_i // K

        cluster_for_split = cluster if not reminder else cluster[:-reminder]

        split = list(maketree(n=N, iterable=cluster_for_split))

        if reminder > 0:
            if reminder_policy == ReminderPolicy.ToLast:
                # extend last.
                split[-1].extend(cluster[-reminder:])
            elif reminder_policy == ReminderPolicy.ToMin:
                min_idx = np.argmin([sum_subsplit_weight(subsplit) for subsplit in split])
                split[min_idx].extend(cluster[-reminder:])
            else:
                raise NotImplementedError()

        all_splits.append(split)
    return all_splits


def make_clusters(graph: Graph, nodes: List[Node], node_weight_function, C: int, THRESHOLD=0):
    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    nodes_below_thresholds = X[X["weight"] <= THRESHOLD]

    # filter below-threashold items
    nodes_above_thresholds = X[X["weight"] > THRESHOLD]

    kmeans = KMeans(n_clusters=C, max_iter=1000, copy_x=True).fit(nodes_above_thresholds)
    X.loc[nodes_above_thresholds.index, "cluster"] = kmeans.labels_

    # For nodes with zero weight, just unify them to close "real" node.
    to_unify = defaultdict(list)

    set_idx = set(nodes_below_thresholds.index)
    for node_id in nodes_below_thresholds.index:
        node = graph[node_id]
        for y in node.out_edges:
            if y.id not in set_idx:
                dst_cluster = X.loc[y.id]['cluster']
                X.loc[X.index == node_id, 'cluster'] = dst_cluster
                to_unify[dst_cluster].append([node_id, y.id])
                break
            # else:
            #     print(f"node_id:{node_id}, y.id:{y.id}", node.scope, y.scope)
        else:
            # Unify:
            print(f"Going 1 more nesting level for node:{node_id} because all outputs are in {set_idx}")

            broke = False

            def _basic_nest(y, X, set_idx, node_id, to_unify):
                for y in y.out_edges:  # This is the nesting level
                    if y.id not in set_idx:
                        dst_cluster = X.loc[y.id]['cluster']
                        X.loc[X.index == node_id, 'cluster'] = dst_cluster
                        to_unify[dst_cluster].append([node_id, y.id])
                        return True
                return False

            for y in node.out_edges:
                broke = _basic_nest(y, X, set_idx, node_id, to_unify)  # 1
                if broke:
                    break
            else:
                # while not broke:
                for y in node.out_edges:
                    for y in y.out_edges:
                        broke = _basic_nest(y, X, set_idx, node_id, to_unify)  # 2
                        if broke:
                            break
                    if broke:
                        break
                else:
                    for y in node.out_edges:
                        for y in y.out_edges:
                            for y in y.out_edges:
                                broke = _basic_nest(y, X, set_idx, node_id, to_unify)  # 3
                                if broke:
                                    break
                            if broke:
                                break
                        if broke:
                            break

            if not broke:
                raise NotImplementedError(f"need to go one level deeper "
                                          f"to find node with above THRESHOLD={THRESHOLD} weight to unify")

    # sort clusters by id.
    cluster_to_min_node_id = {i: X.query(f"cluster == {i}").first_valid_index() for i in range(C)}
    min_node_id_to_cluster = {v: i for i, v in cluster_to_min_node_id.items()}
    Y = X.copy()
    for i, (min_node_id, cluster_id) in enumerate(min_node_id_to_cluster.items()):
        Y.loc[X["cluster"] == cluster_id, 'cluster'] = i

    # Print
    X = Y
    print(X)
    Y = X.copy()
    Y['scope'] = [node.scope for node in nodes]
    print(Y)
    cluster_sums = X.groupby("cluster")['weight'].describe().transpose()
    print("cluster_sums_statistics", cluster_sums)

    clusters = X

    return clusters, to_unify


def first_fit_cluster(K: int, clusters, id_to_node: Dict[int, Node],
                      to_unify: Dict[int, List[Union[List, Any]]], C: int,
                      second_and_on_cluster_policy: SecondAndOnClusterPolicy = SecondAndOnClusterPolicy.FirstFitBinPacking
                      ):
    # result
    bins = defaultdict(list)
    bin_weights = heapdict({i: 0 for i in range(K)})
    # get splits
    all_splits = get_all_splits(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C)

    def choose_bin(subsplit, subsplit_idx, cluster_idx):
        # TODO: be smarter after the 1st cluster

        if cluster_idx == 0 and subsplit_idx >= K:
            warnings.warn(
                f"not fully implemented behavior for 1st cluster subsplit_idx >= K (subsplit_idx:{subsplit_idx},K:{K}), will do FirstFitBinPacking")
        if cluster_idx == 0 and subsplit_idx < K:
            return subsplit_idx
        elif second_and_on_cluster_policy == SecondAndOnClusterPolicy.FirstFitBinPacking or (
                cluster_idx == 0 and subsplit_idx >= K):
            # Tradeoff: communication vs computational balance.
            # Choose bin with minimal weight.
            # subsplits are given by size, decreasing order
            (emptiest_bin_id, current_bin_weight) = bin_weights.peekitem()
            return emptiest_bin_id
        elif second_and_on_cluster_policy == SecondAndOnClusterPolicy.InOrder:
            if subsplit_idx < K:
                return subsplit_idx
            else:
                raise NotImplementedError("probably 1st cluster  >= K came here")
        elif second_and_on_cluster_policy == SecondAndOnClusterPolicy.Reversed:
            if subsplit_idx < K:
                if cluster_idx % 2 != 0:
                    return K - subsplit_idx - 1  # reversed
                else:
                    return subsplit_idx
            else:
                raise NotImplementedError("probably 1st cluster >= K came here")

        #     # if cluster_idx % 2 != 0:
        #     #     return K - subsplit_idx - 1  # reversed
        # return subsplit_idx

    # Partition splits in bins
    for cluster_idx, split in enumerate(reversed(all_splits)):
        # larger clusters will be first
        if len(split) > K and cluster_idx == 0:
            raise NotImplementedError()

        if cluster_idx > 0 and second_and_on_cluster_policy == SecondAndOnClusterPolicy.FirstFitBinPacking:
            # sort subsplits by size
            split = sorted(split, key=sum_subsplit_weight, reverse=True)

        for subsplit_idx, subsplit in enumerate(split):
            bin_idx = choose_bin(subsplit, subsplit_idx, cluster_idx)
            bin = bins[bin_idx]
            to_add = flatten_subsplit(subsplit)
            bin.extend(to_add)
            bin_weights[bin_idx] += sum(i.weight for i in to_add)

    assert len(bins) == K

    return bins


def stages_from_bins(graph, bins, id_to_node_worked_on):
    stage_id_generator = count()

    # shallow copy bins:
    bins_to_id = {i: set(n.id for n in v) for i, v in bins.items()}

    nodes_with_out_edges = defaultdict(set)
    nodes_with_in_edges = defaultdict(set)
    for gpu_id, v in bins.items():
        # Find all connected components
        uf = UnionFind(elements=bins_to_id[gpu_id])
        visited = set()
        open = deque(sorted(v, key=lambda x: x.id))
        while open:
            x = open.popleft()
            x: Node
            for y in x.out_edges:
                if y.id not in uf:
                    nodes_with_out_edges[gpu_id].add(x)
                    nodes_with_in_edges[y.gpu_id].add(y)
                    continue
                uf.union(x.id, y.id)

        # Now, it is problematic if we have:
        #  a->d, b->d, c->d, b->c, and b->d
        # each on different gpu.
        # problem is we can't say {b,d} are same stage because it will create a cycle:
        # a->bd->c->bd
        # if we can break bd, we can solve it afterwards.
        # we already know how to break:
        # af->b->c->d->e->af
        # but how can we know it?

        unbroken_stages = uf.sorted_components()
        broken_stages = []
        # Break stages according to topological sort
        for unbroken_stage in unbroken_stages:
            broken_stages_for_unbroken_stage = []
            cur_set = list()  # its sorted so its more efficient
            unbroken_stage = deque(sorted(unbroken_stage))
            prev_topo_sort_id = unbroken_stage.popleft()
            cur_set.append(prev_topo_sort_id)
            while unbroken_stage:
                topo_sort_id = unbroken_stage.popleft()
                if topo_sort_id == prev_topo_sort_id + 1:
                    # easy, nothing to do.
                    pass
                else:
                    # Check if nothing is missing (it is possible due to layers_graph)
                    missing_topo_sort_ids = range(prev_topo_sort_id + 1, topo_sort_id)
                    is_ok = True
                    for missing_topo_sort_id in missing_topo_sort_ids:
                        # TODO: missing_topo_sort_id in graph is redundant, but here just in case
                        if missing_topo_sort_id in graph and missing_topo_sort_id in id_to_node_worked_on:
                            is_ok = False
                            break
                    if not is_ok:
                        broken_stages_for_unbroken_stage.append(cur_set)
                        cur_set = list()
                    else:  # Nothing is missing
                        cur_set.append(topo_sort_id)

                cur_set.append(topo_sort_id)
                prev_topo_sort_id = topo_sort_id

            if cur_set:
                broken_stages_for_unbroken_stage.append(cur_set)
            broken_stages.extend(broken_stages_for_unbroken_stage)

        broken_stages.sort(key=lambda topo_sorted_list: topo_sorted_list[0])

        print("unbroken_stages")
        print(unbroken_stages)
        print("broken_stages")
        print(broken_stages)

        # Give a dummy stage id
        for dummy_stage_id, broken_stage in zip(stage_id_generator, broken_stages):
            # Try to break stage if not TOPOLOGICALLY sorted.
            for n in broken_stage:
                graph[n].stage_id = dummy_stage_id

    # cannonize_partition_indices(graph) <--- TODO: redundant
    # break cycles <--- TODO: redundant


def analyze_n_clusters(nodes: List[Node], node_weight_function, max_k=10):
    """ utility to help determine number of clusters for partition_2dbin_pack"""

    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    print(X)
    Y = X.copy()
    Y['scope'] = [node.scope for node in nodes]
    print(Y)

    sse = {}
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
        X["cluster"] = kmeans.labels_
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()
    n_clusters = input("choose desired number of clusters to continue...")
    n_clusters = int(n_clusters)
    return n_clusters


def partition_2dbin_pack(graph: Graph,
                         num_gpus: int,
                         n_clusters: int,
                         node_weight_function: Optional[NodeWeightFunction] = None,
                         # edge_weight_function: Optional[EdgeWeightFunction] = None,
                         use_layers_graph: bool = True,
                         THRESHOLD=0,
                         second_and_on_cluster_policy: SecondAndOnClusterPolicy = SecondAndOnClusterPolicy.FirstFitBinPacking,
                         **kwargs
                         ):
    print(f"use_layers_graph={use_layers_graph}")
    graph.topo_sort()

    if use_layers_graph:
        work_graph, lookup = graph.layers_graph()
    else:
        work_graph, lookup = graph, None

    nodes = [n for n in work_graph.nodes if n not in work_graph.inputs]

    K = num_gpus
    if "analyze_n_clusters" in kwargs and kwargs["analyze_n_clusters"]:
        n_clusters = analyze_n_clusters(nodes, node_weight_function, max_k=10)
    # import sys
    # sys.exit(0)
    id_to_node = {node.id: node for node in nodes}
    C = n_clusters

    clusters, to_unify = make_clusters(work_graph, nodes, node_weight_function, C=C, THRESHOLD=THRESHOLD)
    bins = first_fit_cluster(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C,
                             second_and_on_cluster_policy=second_and_on_cluster_policy)
    # sort
    for v in bins.values():
        v.sort(key=lambda x: x.Index)
    pprint(bins)

    # To nodes list
    def node_list(iterable):
        return [id_to_node[i.Index] for i in iterable]

    bins = {i: node_list(bins[i]) for i in bins}
    # Balance:
    times = {i: sum(node_weight_function(x) for x in bins[i]) for i in bins}
    print("times:")
    pprint(times)

    # Bins to GPUs:
    for i, bin_nodes in bins.items():
        for n in bin_nodes:
            n.gpu_id = i

    # bins to stages
    stages_from_bins(work_graph, bins, id_to_node_worked_on=id_to_node)

    work_graph = post_process_partition(work_graph)

    if use_layers_graph:
        graph.induce_layer_partition(work_graph, lookup)

    # Get stage to GPU map
    node_to_stage_map = {}
    # Convert
    stage_to_gpu_map = defaultdict(set)
    for gpu_id, bin_nodes in bins.items():
        for n in bin_nodes:
            n: Node
            stage_to_gpu_map[n.stage_id].add(gpu_id)
            node_to_stage_map[n.id] = n.stage_id

    stage_to_gpu_map = {i: sorted(v) for i, v in stage_to_gpu_map.items()}

    # TODO: can do it more efficiently but i'm tired...
    to_check = sorted(stage_to_gpu_map.keys())
    if to_check[0] != 0 or to_check[-1] != len(to_check) - 1:
        print(f"-V- stages gone, stages_ids_before: {to_check} reassigning...")

        stage_to_fixed = {prev_s: i for i, prev_s in enumerate(to_check)}

        # 1
        for n, prev_s in list(node_to_stage_map.items()):
            fix = stage_to_fixed[prev_s]
            node_to_stage_map[n] = fix

        # 2
        for n in graph.nodes:
            n.stage_id = stage_to_fixed[n.stage_id]

        # 3
        stage_to_gpu_map = defaultdict(set)
        for gpu_id, bin_nodes in bins.items():
            for n in bin_nodes:
                stage_to_gpu_map[n.stage_id].add(gpu_id)
        stage_to_gpu_map = {i: sorted(v) for i, v in stage_to_gpu_map.items()}

    print("stage_to_gpu_map:")
    pprint(stage_to_gpu_map)

    print("node_to_stage_map:")
    pprint(node_to_stage_map)

    stage_to_nodes_map = defaultdict(list)
    for i, v in node_to_stage_map.items():
        stage_to_nodes_map[v].append(i)

    print("stage_to_nodes_map:")
    pprint(stage_to_nodes_map)

    return graph, stage_to_gpu_map


if __name__ == '__main__':
    from pytorch_Gpipe import build_graph
    import torch
    from torch.nn import Sequential, Linear

    IN_FEATURES = 320
    OUT_FEATURES = 8

    model = Sequential(
        *[Linear(IN_FEATURES, IN_FEATURES), Linear(IN_FEATURES, IN_FEATURES), Linear(IN_FEATURES, IN_FEATURES), Linear(
            IN_FEATURES, IN_FEATURES),
          Linear(IN_FEATURES, OUT_FEATURES),
          Linear(OUT_FEATURES, OUT_FEATURES), Linear(OUT_FEATURES, OUT_FEATURES), Linear(OUT_FEATURES, OUT_FEATURES)])

    inputs = torch.randn(IN_FEATURES, IN_FEATURES)

    model = model.cuda()
    inputs = inputs.cuda()
    graph = build_graph(model, args=(inputs,), n_iter=50)

    node_weight_function = NodeWeightFunction(bwd_to_fwd_ratio=1, MULT_FACTOR=100000)
    # graph.display(node_weight_function=node_weight_function)
    # dot = graph.build_dot(node_weight_function=node_weight_function)
    # graphviz.Source(graph.build_dot(node_weight_function=node_weight_function))
    # nxg = graph.asNetworkx(directed=False, node_weight_function=node_weight_function)
    # import matplotlib.pyplot as plt
    # nx.draw_networkx(nxg, labels={n: {"weight": v["weight"]} for n,v in nxg.nodes.items()})
    # plt.show()

    nodes = [n for n in graph.nodes if n not in graph.inputs]
    # analyze_n_clusters(nodes=nodes, node_weight_function=node_weight_function, max_k=4)
    graph, stage_to_gpu_map = partition_2dbin_pack(graph=graph, num_gpus=2, n_clusters=2,
                                                   node_weight_function=node_weight_function,
                                                   use_layers_graph=False)

# TODO: Actually do the bin pack (first fit) with the splits (triples).
#    Starting from 2nd cluster it matters.
#    However, it can cause weird communication pattern.
# TODO: weird error: "keyerror 0" - stage disappeared! (can fix it afterwards)
# TODO: THRESHOLD hyper-parameter can be higher.

#################
# Layers for T5 small (tied)
#     LAYER_SCOPES=[
# >>            'T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]',
# >>            'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]',
#               'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
# >>            'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]',
# >>            'T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm]',
#               'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
#               'T5ForConditionalGeneration/Linear[lm_head]',
#               'T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]',
#           ]
#################
#################
#################

# when partitioning to 2 gpus we want something like:
#     LAYER_SCOPES_TO_GPU={
# >>            'T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens]':0,
#               'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]':0,
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]':0,
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]':0,
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]':0,
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]':1,
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]':1,
#               'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]':1,
# >>            'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]':1,
#               'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]':1,
# >>            'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]':1,
#               'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]':1,
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]':1,
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]':1,
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]':1,
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]':0,
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]':0,
#               'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]':0,
# >>            'T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm]':0,
#               'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]':0,
#               'T5ForConditionalGeneration/Linear[lm_head]':0,
#               'T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]':0,
#           }
