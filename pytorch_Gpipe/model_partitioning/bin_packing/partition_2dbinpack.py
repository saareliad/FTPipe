import warnings
from collections import deque, defaultdict
from itertools import count, zip_longest
from pprint import pprint
from typing import Optional, List, Dict, Any, Union

import matplotlib.pyplot as plt
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


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_all_splits(K: int, clusters, id_to_node: Dict[int, Node], to_unify: Dict[int, List[Union[List, Any]]], C: int):
    all_splits = []

    # TODO: unify

    # for group in to_unify_for_cluster:

    # Make splits
    # Change data type
    # Pandas object. (id->Index)

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
            warnings.warn(f"cluster {c_i} is problematic {c_i}, {n_i}%{K}!=0, "
                          f"will put reminding {reminder} nodes in last partition")
            # raise NotImplementedError(f"{c_i}, {n_i}, {K}")
        N = n_i // K

        cluster_for_split = cluster if not reminder else cluster[:-reminder]
        split = list(grouper(n=N, iterable=cluster_for_split))

        if reminder > 0:
            # extend last.
            # FIXME:
            split[-1].extend(cluster[-reminder:])

        all_splits.append(split)
    return all_splits


def make_clusters(graph: Graph, nodes: List[Node], node_weight_function, C: int, THRESHOLD=0):
    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    kmeans = KMeans(n_clusters=C, max_iter=1000).fit(X)
    X["cluster"] = kmeans.labels_

    # TODO: for nodes with zero weight, just unify them to close node.
    to_unify = defaultdict(list)
    nodes_below_thresholds = X[X["weight"] <= THRESHOLD]

    set_idx = set(nodes_below_thresholds.index)
    for node_id in nodes_below_thresholds.index:
        node = graph[node_id]
        for y in node.out_edges:
            if y.id not in set_idx:
                dst_cluster = X.iloc[y.id]['cluster']
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
                        dst_cluster = X.iloc[y.id]['cluster']
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
                else:
                    for y in node.out_edges:
                        for y in y.out_edges:
                            for y in y.out_edges:
                                broke = _basic_nest(y, X, set_idx, node_id, to_unify)  # 3
                                if broke:
                                    break
            if not broke:
                raise NotImplementedError(f"need to go one level deeper "
                                          f"to find node with above THRESHOLD={THRESHOLD} weight to unify")

    # sort clusters by id.
    cluster_to_min_node_id = {i: X.query(f"cluster == {i}").first_valid_index() for i in range(C)}
    min_node_id_to_cluster = {v: i for i, v in cluster_to_min_node_id.items()}
    Y = X.copy()
    for i, v in enumerate(min_node_id_to_cluster):
        Y[X["cluster"] == v]['cluster'] = i

    # Print
    X = Y
    print(X)
    Y = X.copy()
    Y['scope'] = [node.scope for node in nodes]
    print(Y)
    cluster_sums = X.groupby("cluster")['weight'].sum()
    print("cluster_sums", cluster_sums)

    clusters = X

    return clusters, to_unify


def first_fit_cluster(K: int, clusters, id_to_node: Dict[int, Node],
                      to_unify: Dict[int, List[Union[List, Any]]], C: int):
    # result
    bins = defaultdict(list)
    bin_weights = heapdict({i: 0 for i in range(K)})
    # get splits
    all_splits = get_all_splits(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C)

    def choose_bin(subsplit, subsplit_idx, cluster_idx):
        # TODO: be smarter after the 1st cluster
        if cluster_idx % 2 != 0:
            return K - subsplit_idx - 1  # reversed
        return subsplit_idx

    # Partition splits in bins
    for cluster_idx, split in enumerate(all_splits):
        if len(split) != K:
            raise NotImplementedError()

        for subsplit_idx, subsplit in enumerate(split):
            bin_idx = choose_bin(subsplit, subsplit_idx, cluster_idx)
            bin = bins[bin_idx]
            to_add = []
            for i in subsplit:
                if isinstance(i, list):
                    to_add.extend(i)
                else:
                    to_add.append(i)
            # to_add = [i for i in subsplit if is]
            # to_add = list(flatten(subsplit))
            bin.extend(to_add)
            bin_weights[bin_idx] += sum(i.weight for i in to_add)

    assert len(bins) == K

    return bins


def stages_from_bins(graph, bins):
    stage_id_generator = count()

    # shallow copy bins:
    bins_to_id = {i: set(n.id for n in v) for i, v in bins.items()}

    nodes_with_out_edges = defaultdict(set)
    nodes_with_in_edges = defaultdict(set)
    for gpu_id, v in bins.items():
        # TODO: find all connected componenets
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
        # we alreay know how to break:
        # af->b->c->d->e->af
        # but how can we know it?

        # give a dummy stage id
        unbroken_stages = uf.sorted_components()
        for dummy_stage_id, unbroken_stage in zip(stage_id_generator, unbroken_stages):
            for n in unbroken_stage:
                graph[n].stage_id = dummy_stage_id

    # cannonize_partition_indices(graph)
    # break cycles

    # TODO: get max subtree
    # directed DFS, keep ends

    # for end in sorted(ends):

    # get all ends which and on another GPU x

    # output -> another GPU -> create stage


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
                         **kwargs
                         ):
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

    clusters, to_unify = make_clusters(work_graph, nodes, node_weight_function, C=C)
    bins = first_fit_cluster(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C)
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
    stages_from_bins(work_graph, bins)

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
    print("stage_to_gpu_map:")
    pprint(stage_to_gpu_map)

    print("node_to_stage_map:")
    pprint(node_to_stage_map)

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
                                                   node_weight_function=node_weight_function)

# TODO: weird "cycle". seems that just switching will solve
# [10, 4]
# [(30, 10, 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[17]'), (31, 4, 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[18]')]
# n_partitions: 20
# TODO consider doing another cluster for zeros weights and so on.
# TODO: support smaller cluster
# TODO: Actually do the bin pack (first fit) with the splits (triples).
#    Starting from 2nd cluster it matters.
#    However, it can cause weird communication pattern.
# TODO: weird error: "keyerror 0" - stage disappeared! (can fix it afterwards)
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
