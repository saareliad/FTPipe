from collections import deque, defaultdict
from pprint import pprint
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from pytorch_Gpipe.model_partitioning.heuristics import NodeWeightFunction
from pytorch_Gpipe.model_profiling import Graph, Node


# from ...model_profiling import Graph, Node, NodeWeightFunction


# from .partition_2dbinpack_poc import determine_n_clusters, first_fit_cluster, make_clusters
# Poc abstraction for 2d-packing for pipeline with virtual stages
# ######## ######### ############ ############# ########## ########### ####### ############
# sorts nodes by value
# cluster into similar sizes with n_1, n_2, ..., n_C sizes
# for each cluster i=1...C:
# sort by sequential (id)
# take N =  n_i // K values, and maketree.
# ######## ######## ############# ############## ######### ########### ######## ###########


def maketree(iterable, N):
    d = deque(iterable)
    res = []
    while d:
        pair = [d.popleft() for _ in range(N)]
        res.append(pair)
    return res


def first_fit_cluster(K, clusters):
    if len(clusters) > 2:
        raise NotImplementedError()

    # def add_id_to_split_k(list_of_items):
    #     for

    # result
    bins = defaultdict(list)

    # get splits
    all_splits = get_all_splits(K, clusters)

    # Unify splits to bins
    for k in range(K):
        for split in all_splits:
            assert len(split) == K, (len(split), len(split[k]), split)
            print(split[k])
            bins[k].extend(split[k])

    assert len(bins) == K

    return bins


def get_all_splits(K, clusters):
    all_splits = []
    for c_i, cluster in enumerate(clusters):
        n_i = len(cluster)
        if n_i < K:
            raise NotImplementedError(f"{c_i}, {n_i}, {K}")
        if n_i % K != 0:
            raise NotImplementedError(f"{c_i}, {n_i}, {K}")
        N = n_i // K
        split = maketree(cluster, N=N)
        is_reversed = c_i % 2 == 0
        if is_reversed:
            split = list(reversed(split))
        all_splits.append(split)
    return all_splits


def make_clusters(nodes: List[Node], node_weight_function, C: int):
    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    kmeans = KMeans(n_clusters=C, max_iter=1000).fit(X)
    X["cluster"] = kmeans.labels_

    return X


def determine_n_clusters(nodes: List[Node], node_weight_function, max_k=10):
    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    # print(X)
    sse = {}
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
        X["cluster"] = kmeans.labels_
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()


def partition_2dbin_pack(graph: Graph,
                         num_gpus: int,
                         n_clusters: int,
                         node_weight_function: Optional[NodeWeightFunction] = None,
                         # edge_weight_function: Optional[EdgeWeightFunction] = None,
                         ):
    K = num_gpus
    nodes = graph.nodes
    # determine_n_clusters(nodes, node_weight_function, max_k=10)
    # import sys
    # sys.exit(0)
    id_to_node = {node.id: node for node in nodes}
    C = n_clusters
    X = make_clusters(nodes, node_weight_function, C=C)
    # print(X)
    cluster_sums = X.groupby("cluster")['weight'].sum()
    print("cluster_sums", cluster_sums)
    # Pandas object. (id->Index)
    clusters = [list(X.groupby("cluster").get_group(c).sort_values("id").itertuples()) for c in range(C)]
    clusters_lengths = {i: len(clusters[i]) for i in range(len(clusters))}
    print("cluster_lengths", clusters_lengths)
    bins = first_fit_cluster(K, clusters)
    # sort
    for v in bins.values():
        v.sort(key=lambda x: x.Index)
    pprint(bins)

    # To nodes list
    def node_list(iterable):
        return [id_to_node[i.Index] for i in iterable]

    bins = {i: node_list(bins[i]) for i in bins}
    # Balance:
    times = {i: sum(x.weight for x in bins[i]) for i in bins}
    print("times:")
    pprint(times)

    # Convert
    for stage_id, ns in bins:
        for n in ns:
            n: Node
            n.stage_id = stage_id

    return graph


if __name__ == '__main__':
    from pytorch_Gpipe import build_graph

    from torch.nn import Sequential, Linear
    import torch

    model = Sequential(
        *[Linear(10, 10), Linear(10, 10), Linear(10, 10), Linear(10, 10),
          Linear(10, 5),
          Linear(5, 5), Linear(5, 5), Linear(5, 5)])
    graph = build_graph(model, args=torch.randn(10, 10))
    node_weight_function = NodeWeightFunction(bwd_to_fwd_ratio=1)
    partition_2dbin_pack(graph=graph, num_gpus=2, n_clusters=2, node_weight_function=node_weight_function)
