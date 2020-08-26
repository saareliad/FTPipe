from collections import deque, defaultdict
from itertools import count
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
# Poc abstraction for 2d-packing for pipeline with virtual stages
# ######## ######### ############ ############# ########## ########### ####### ############
# sorts nodes by value
# cluster into similar sizes with n_1, n_2, ..., n_C sizes
# for each cluster i=1...C:
# sort by sequential (id)
# take N =  n_i // K values, and maketree.
# ######## ######## ############# ############## ######### ########### ######## ###########


def maketree(iterable, N, K):
    # print("maketree", N, K)
    d = deque(iterable)
    res = []
    while d:
        pair = [d.popleft() for _ in range(N)]
        res.append(pair)
    return res


def first_fit_cluster(K, clusters):
    if len(clusters) > 2:
        raise NotImplementedError()

    # result
    bins = defaultdict(list)

    # get splits
    all_splits = []
    for c_i, cluster in enumerate(clusters):
        n_i = len(cluster)
        if n_i < K:
            raise NotImplementedError(f"{c_i}, {n_i}, {K}")
        if n_i % K != 0:
            raise NotImplementedError(f"{c_i}, {n_i}, {K}")
        N = n_i // K
        # print(f"{c_i}, {n_i}, {K}, {N}")
        split = maketree(cluster, N=N, K=K)
        # print(len(split))
        # print(len(split[0]))
        is_reversed = c_i % 2 == 0
        if is_reversed:
            split = list(reversed(split))
        all_splits.append(split)

    # Unify splits to bins
    for k in range(K):
        for split in all_splits:
            assert len(split) == K, (len(split), len(split[k]), split)
            bins[k].extend(split[k])

    assert len(bins) == K

    return bins


def make_clusters(nodes, C):
    def node_to_record(node):
        return {"id": node.id, "w": node.w}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    kmeans = KMeans(n_clusters=C, max_iter=1000).fit(X)
    X["cluster"] = kmeans.labels_

    return X


def determine_n_clusters(nodes, max_k=10):
    def node_to_record(node):
        return {"id": node.id, "w": node.w}

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
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


if __name__ == '__main__':
    K = 8


    class Node:
        def __init__(self, id, w):
            self.w = w
            self.id = id


    c = count()
    A = [Node(next(c), 9) for _ in range(24)]
    B = [Node(next(c), 2) for _ in range(24)]

    nodes = [*A, *B]

    # determine_n_clusters(nodes, max_k=4)
    # import sys
    # sys.exit(0)

    id_to_node = {node.id: node for node in nodes}

    C = 2
    X = make_clusters(nodes, C=C)

    # print(X)

    cluster_sums = X.groupby("cluster")['w'].sum()
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
    times = {i: sum(x.w for x in bins[i]) for i in bins}
    print("times:")
    pprint(times)
