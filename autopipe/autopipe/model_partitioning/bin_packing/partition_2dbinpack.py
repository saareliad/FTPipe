import warnings
from collections import deque, defaultdict
from enum import Enum
from itertools import count
from pprint import pprint
from typing import Optional, List, Dict, Any, Union, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from autopipe.autopipe.model_partitioning.bin_packing.heap_dict import heapdict
from autopipe.autopipe.model_partitioning.bin_packing.post_process import post_process_partition
from autopipe.autopipe.model_partitioning.bin_packing.union_find import UnionFind
from autopipe.autopipe.model_partitioning.heuristics import NodeWeightFunction
from autopipe.autopipe.model_profiling import Graph, Node


# from ...model_profiling import Graph, Node, NodeWeightFunction


# from .partition_2dbinpack_poc import analyze_n_clusters, best_Fit_cluster, make_clusters
# Poc abstraction for 2d-packing for pipeline with virtual stages
# ######## ######### ############ ############# ########## ########### ####### ############
# sorts nodes by value
# cluster into similar sizes with n_1, n_2, ..., n_C sizes
# for each cluster i=1...C:
# sort by sequential (id)
# take N =  n_i // K values, and maketree.
# ######## ######## ############# ############## ######### ########### ######## ###########

class ReminderPolicy(Enum):
    ToLast = "last"
    ToMin = "min"


class SecondAndOnClusterPolicy(Enum):
    BestFitBinPacking = "best_fit"
    InOrder = "order"
    Reversed = "reversed"


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
    assert len(clusters.cluster.unique()) == C
    # Make splits
    # Change data type: Pandas object. (id->Index)
    clusters = [list(clusters.groupby("cluster").get_group(c).sort_values("id").itertuples()) for c in range(C)]
    # clusters: List[List[Any]]
    clusters_lengths = {i: len(clusters[i]) for i in range(len(clusters))}
    print("cluster_lengths, before compressing", clusters_lengths)
    # nest clusters with zero
    new_clusters = get_unified_clusters(clusters, to_unify)

    clusters = new_clusters

    clusters_lengths = {i: len(clusters[i]) for i in range(len(clusters))}
    print("cluster_lengths, after compressing", clusters_lengths)

    for c_i, cluster in enumerate(clusters):
        n_i = len(cluster)
        reminder = n_i % K
        only_reminder = False
        if n_i < K:
            only_reminder = True
            warnings.warn(
                f"small number of items in cluster {c_i}: {n_i} need at least {K}. will treat as reminder with policy {reminder_policy}")
            # raise NotImplementedError(f"insufficient number of items in cluster {c_i}: {n_i} need at least {K}")
        # if reminder > 0:
        #     pass

        N = n_i // K

        if not only_reminder:
            cluster_for_split = cluster if not reminder else cluster[:-reminder]
            split = list(maketree(n=N, iterable=cluster_for_split))
        else:
            # split = [[] for _ in range(K)]
            cluster_for_split = cluster
            split = [[x] for x in cluster_for_split]
            reminder = 0

        if reminder > 0:
            if reminder_policy == ReminderPolicy.ToLast:
                warnings.warn(
                    f"cluster {c_i} is problematic, {n_i}%{K}!=0, will put reminding {reminder} nodes in last partition")
                # extend last.
                split[-1].extend(cluster[-reminder:])
            elif reminder_policy == ReminderPolicy.ToMin:
                warnings.warn(
                    f"cluster {c_i} is problematic {c_i}, {n_i}%{K}!=0, will put reminding {reminder} nodes in min weight partition")
                min_idx = np.argmin([sum_subsplit_weight(subsplit) for subsplit in split])
                split[min_idx].extend(cluster[-reminder:])
            else:
                raise NotImplementedError(f"reminder_policy:{reminder_policy}")

        all_splits.append(split)
    return all_splits


def get_unified_clusters(clusters, to_unify):
    # init state used to check
    def to_set(v, s):
        if not isinstance(v, list):
            s.add(v)
            return
        for x in v:
            to_set(x, s)

    A, B = set(), set()
    to_set(clusters, A)

    new_clusters = []
    for c_i, cluster in enumerate(clusters):
        to_unify_for_cluster = to_unify[c_i]
        cluster_D = {i.Index: i for i in cluster}
        deleted_from_cluster = {}  # dict: deleted->new_dest
        next_gen = []
        first_time = True
        while first_time or next_gen:
            first_time = False
            if next_gen:
                to_unify_for_cluster = next_gen
                next_gen = []

            for l in to_unify_for_cluster:
                # make a list
                # before:   d: id->item
                # after:    d: id->[item, item]
                z = l[0]
                x = l[1]
                if isinstance(cluster_D[x], list):
                    # already was listed
                    v = cluster_D[x]
                    zz = cluster_D[z]
                    if isinstance(zz, list):
                        v.extend(zz)
                    else:
                        v.append(zz)
                    v.sort(key=lambda y: y.Index)
                else:
                    # Make a list and put it
                    v = [cluster_D[x]]
                    try:
                        zz = cluster_D[z]
                    except KeyError as e:
                        if not z in deleted_from_cluster:
                            raise e
                        else:
                            warnings.warn(
                                f"found a double: {l}, I already deleted {z} and unified it with {deleted_from_cluster[z]}, will unify now to {x}")
                            next_gen.append(sorted([x, deleted_from_cluster[z]]))
                            continue

                    if isinstance(zz, list):
                        v.extend(zz)
                    else:
                        v.append(zz)
                    v.sort(key=lambda y: y.Index)
                    cluster_D[x] = v

                # delete what we unified.
                deleted_from_cluster[z] = x
                del cluster_D[z]

        cluster = []
        for i in sorted(cluster_D.keys()):
            cluster.append(cluster_D[i])

        new_clusters.append(cluster)

    to_set(new_clusters, B)
    assert A == B, (A, B)
    return new_clusters


def make_clusters(graph: Graph, nodes: List[Node], node_weight_function, C: int, THRESHOLD=0):
    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    nodes_below_thresholds = X[X["weight"] <= THRESHOLD]

    # filter below-threshold items
    nodes_above_thresholds = X[X["weight"] > THRESHOLD]

    kmeans = KMeans(n_clusters=C, max_iter=1000, copy_x=True).fit(nodes_above_thresholds)
    X.loc[nodes_above_thresholds.index, "cluster"] = kmeans.labels_

    # For nodes with zero weight, just unify them to close "real" node.
    to_unify = defaultdict(list)
    set_idx = set(nodes_below_thresholds.index)

    def _basic_nest_forward(y: Node, X, set_idx, node_id, to_unify, nesting_to_take=0, curr_nesting=0):
        for y in y.out_edges:  # This is the nesting level
            if curr_nesting == nesting_to_take:
                if y.id not in set_idx:
                    dst_cluster = X.loc[y.id]['cluster']
                    X.loc[X.index == node_id, 'cluster'] = dst_cluster
                    to_unify[dst_cluster].append([node_id, y.id])
                    print(f"-V- unify node: {node.id} to dst: {y.id}, cluster: {dst_cluster} (forward)")
                    return True
            else:
                return _basic_nest_forward(y, X, set_idx, node_id, to_unify, nesting_to_take=nesting_to_take,
                                           curr_nesting=curr_nesting + 1)
        return False

    def _basic_nest_backward(y: Node, X, set_idx, node_id, to_unify, nesting_to_take=0, curr_nesting=0):
        for y in y.in_edges:  # This is the nesting level
            if curr_nesting == nesting_to_take:
                if y.id not in set_idx:
                    try:
                        dst_cluster = X.loc[y.id]['cluster']
                    except KeyError:
                        continue
                    X.loc[X.index == node_id, 'cluster'] = dst_cluster
                    to_unify[dst_cluster].append([y.id, node_id])  # intentional: reverse
                    print(f"-V- unify node: {node.id} to dst: {y.id}, cluster: {dst_cluster} (backward)")
                    return True
            else:
                return _basic_nest_backward(y, X, set_idx, node_id, to_unify, nesting_to_take=nesting_to_take,
                                            curr_nesting=curr_nesting + 1)
        return False

    for node_id in nodes_below_thresholds.index:
        node = graph[node_id]
        # find a "big"
        # go over candidates by distance
        # Note: this can be implemented with graph matirx representation and mamtuls
        # (A^x length x paths)
        nesting_to_take = 0
        NESTING_LIMIT = len(graph) + 1
        while True:

            broke = _basic_nest_forward(node, X, set_idx, node_id, to_unify, nesting_to_take=nesting_to_take,
                                        curr_nesting=0) or _basic_nest_backward(node, X, set_idx, node_id, to_unify,
                                                                                nesting_to_take=nesting_to_take,
                                                                                curr_nesting=0)
            if broke:
                break

            nesting_to_take += 1
            print(
                f"Going {nesting_to_take + 1} more nesting level for node:{node_id} because all outputs are below threshold {set_idx}")
            if nesting_to_take >= NESTING_LIMIT:
                raise NotImplementedError(f"did not find node with above THRESHOLD={THRESHOLD} weight to unify")

    # fix to 8->9 and 8->10 and 9->10 problem which is reduced to 8->9->10 and useless data jump between gpus

    # topo sort out edges for nodes: (works since the graph is topo sorted)
    for n in graph.nodes:
        n.out_edges.sort(key=lambda x: x.id)

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


def best_Fit_cluster(K: int, clusters, id_to_node: Dict[int, Node],
                     to_unify: Dict[int, List[Union[List, Any]]], C: int,
                     second_and_on_cluster_policy: SecondAndOnClusterPolicy = SecondAndOnClusterPolicy.BestFitBinPacking,
                     reminder_policy: ReminderPolicy = ReminderPolicy.ToLast,
                     ):
    # FIXME: some bins are missing!
    # result
    bins = defaultdict(list)
    bin_weights = heapdict({i: 0 for i in range(K)})
    bin_memory = heapdict({i: 0 for i in range(K)})
    # get splits
    all_splits = get_all_splits(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C,
                                reminder_policy=reminder_policy)

    def check_memory_fit(candidate, bin_id):
        # TODO: currently this was only used in PoC for encoder decoder where we know a-priori it will fit.
        return True

    def choose_bin(subsplit, subsplit_idx, cluster_idx):
        # TODO: be smarter after the 1st cluster

        if cluster_idx == 0 and subsplit_idx >= K:
            warnings.warn(
                f"not fully implemented behavior for 1st cluster subsplit_idx >= K (subsplit_idx:{subsplit_idx},K:{K}), will do FirstFitBinPacking")
        if cluster_idx == 0 and subsplit_idx < K:
            return subsplit_idx
        elif second_and_on_cluster_policy == SecondAndOnClusterPolicy.BestFitBinPacking or (
                cluster_idx == 0 and subsplit_idx >= K):
            # Tradeoff: communication vs computational balance.
            # Choose bin with minimal weight.
            # subsplits are given by size, decreasing order
            saved = []
            while True:
                if len(bin_weights) == 0:
                    raise RuntimeError("no bin can fit (memory-wise)")
                (emptiest_bin_id, current_bin_weight) = bin_weights.peekitem()
                fits_memory = check_memory_fit(subsplit, emptiest_bin_id)
                if fits_memory:
                    break
                saved.append(bin_weights.popitem())

            # restore saved
            for k, v in saved:
                bin_weights[k] = v

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

        if len(split) < K and cluster_idx == 0:
            warnings.warn(f"got only reminder in 1st and largest cluster: {split}")

        if cluster_idx > 0 and second_and_on_cluster_policy == SecondAndOnClusterPolicy.BestFitBinPacking:
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


def stages_from_bins(graph: Graph, bins: Dict[int, List[Node]], id_to_node_worked_on: Dict[int, Node]):
    stage_id_generator = count()

    # shallow copy bins, excluding inputs:
    bins_to_id = {i: set(n.id for n in v if n.id in id_to_node_worked_on) for i, v in bins.items()}

    nodes_with_out_edges_to_different_gpu = defaultdict(set)
    nodes_with_in_edges_from_different_gpu = defaultdict(set)
    all_broken_stages = []
    for gpu_id, nodes_in_bin in bins.items():
        # Find all connected components in the bin
        unbroken_stages = get_ccs_on_same_gpu(bins_to_id, gpu_id, nodes_in_bin, nodes_with_in_edges_from_different_gpu,
                                              nodes_with_out_edges_to_different_gpu)

        broken_stages = break_ccs_on_same_gpu_to_stages(graph, id_to_node_worked_on, unbroken_stages, bins_to_id)

        # NOTE: the prints here include inputs. Input stage is arbitrary since it will be changed.
        print("unbroken_stages")
        print(unbroken_stages)
        print("broken_stages")
        print(broken_stages)

        broken_stages: List[List[int]]

        all_broken_stages.extend(broken_stages)

        # # Give a dummy stage id
        # for dummy_stage_id, broken_stage in zip(stage_id_generator, broken_stages):
        #     # Try to break stage if not TOPOLOGICALLY sorted.
        #     for n in broken_stage:
        #         graph[n].stage_id = dummy_stage_id

        # Unify redundant stages
        # re_assign_partition_indices(graph)

    all_broken_stages.sort(key=lambda topo_sorted_list: topo_sorted_list[0])
    for i, topo_sorted_list in enumerate(all_broken_stages):
        for nid in topo_sorted_list:
            n = id_to_node_worked_on[nid]
            n.stage_id = i

    # re_assign_partition_indices(graph) <--- TODO: redundant
    # break cycles <--- TODO: redundant


def break_ccs_on_same_gpu_to_stages(graph, id_to_node_worked_on, unbroken_stages, bins_to_id):
    # Now, it is problematic if we have:
    #  a->d, b->d, c->d, b->c, and b->d
    # each on different gpu.
    # problem is we can't say {b,d} are same stage because it will create a cycle:
    # a->bd->c->bd
    # if we can break bd, we can solve it afterwards.
    # we already know how to break:
    # af->b->c->d->e->af
    # but how can we know it?

    broken_stages = []
    # Break stages according to topological sort
    for unbroken_stage in unbroken_stages:
        broken_stages_for_unbroken_stage = []
        cur_set = list()  # its sorted so its more efficient
        unbroken_stage = deque(sorted(unbroken_stage))

        # Get the first non-input node in stage.
        while unbroken_stage:
            prev_topo_sort_id = unbroken_stage.popleft()
            if prev_topo_sort_id in id_to_node_worked_on:
                cur_set.append(prev_topo_sort_id)
                break
            else:
                print(f"skipping input_v0: {prev_topo_sort_id}")
        # cur_set.append(prev_topo_sort_id)

        while unbroken_stage:
            topo_sort_id = unbroken_stage.popleft()
            if topo_sort_id == prev_topo_sort_id + 1:
                # easy, nothing to do.
                pass
            elif topo_sort_id not in id_to_node_worked_on:
                # a graph input
                print(f"skipping input_v1: {prev_topo_sort_id}")
                # continue
            else:
                # check if there is some path:
                # cur_set --> missing --> {topo_sort_id | unbroken_stage}
                has_path_via_missing_nodes = ccs_on_same_gpu_has_path_via_missing_nodes(cur_set, graph,
                                                                                        id_to_node_worked_on,
                                                                                        prev_topo_sort_id, topo_sort_id,
                                                                                        unbroken_stage)

                if has_path_via_missing_nodes:
                    broken_stages_for_unbroken_stage.append(cur_set)
                    cur_set = list()

                    # FIXME Assert:
                    missing_topo_sort_ids = list(range(prev_topo_sort_id + 1, topo_sort_id))
                    for mid in missing_topo_sort_ids:
                        in_bins = any(map(lambda v: mid in v, bins_to_id.values()))
                        if not in_bins:
                            print(f"missing_topo_sort_ids are not in bins {missing_topo_sort_ids}")
                            raise ValueError(f"missing_topo_sort_ids are not in bins {missing_topo_sort_ids}")

            if topo_sort_id in id_to_node_worked_on:
                cur_set.append(topo_sort_id)
                prev_topo_sort_id = topo_sort_id
            else:
                print(f"skipping input_v2: {prev_topo_sort_id}")
                # Get a new prev_topo_sort_id
                while unbroken_stage:
                    prev_topo_sort_id = unbroken_stage.popleft()
                    if prev_topo_sort_id in id_to_node_worked_on:
                        cur_set.append(prev_topo_sort_id)
                        break
                    else:
                        print(f"skipping input_v3: {prev_topo_sort_id}")

        if cur_set:
            broken_stages_for_unbroken_stage.append(cur_set)
        broken_stages.extend(broken_stages_for_unbroken_stage)
    broken_stages.sort(key=lambda topo_sorted_list: topo_sorted_list[0])
    # #
    # #
    # # for i, topo_sorted_list in enumerate(broken_stages):
    # #     for nid in topo_sorted_list:
    # #         n = id_to_node_worked_on[nid]
    # #         n.stage_id = i

    return broken_stages


def ccs_on_same_gpu_has_path_via_missing_nodes(cur_set, graph, id_to_node_worked_on, prev_topo_sort_id, topo_sort_id,
                                               unbroken_stage):
    # Check if nothing is missing (it is possible due to layers_graph)
    # Example: 9->13
    # prev: {7,8, 9}
    # unbroken: {13,14,15}
    # candidate is 13.
    # Now, the following code
    # checks that there is no path from {7,8,9} to {13,14,15} via missing {10,11,12}
    # if there is: break: 13 can't be together with {7,8,9}.
    missing_topo_sort_ids = list(range(prev_topo_sort_id + 1, topo_sort_id))
    is_ok = True
    for missing_topo_sort_id in missing_topo_sort_ids:
        if not (missing_topo_sort_id in graph and missing_topo_sort_id in id_to_node_worked_on):
            continue
        #  Need to actually check if there is a cycle.
        cur_nodes = [id_to_node_worked_on[x] for x in cur_set]
        scs = set(cur_set)
        missing_nodes_in_work_graph = [id_to_node_worked_on[x] for x in missing_topo_sort_ids if
                                       x not in scs and x in id_to_node_worked_on]
        nodes_left_in_unborken_stage = set(id_to_node_worked_on[x] for x in unbroken_stage)
        nodes_left_in_unborken_stage.add(id_to_node_worked_on[topo_sort_id])

        A: Set[Node] = set(cur_nodes)
        B: Set[Node] = set(nodes_left_in_unborken_stage)
        edge_nodes: Set[Node] = set(missing_nodes_in_work_graph)
        edges = []
        for a in A:
            for c in a.out_edges:
                if c in edge_nodes:
                    edges.append((0, c.id + 2))

        for c in edge_nodes:
            for nc in c.out_edges:
                if nc in edge_nodes:
                    edges.append((c.id + 2, nc.id + 2))
                elif nc in B:
                    edges.append((c.id + 2, 1))

        G = nx.DiGraph(incoming_graph_data=edges)
        G.add_node(0)
        G.add_node(1)
        has_path = nx.algorithms.shortest_paths.generic.has_path(G, 0, 1)
        # Scream if has path
        is_ok = not has_path
        if not is_ok:
            break
    has_path_via_missing_nodes = not is_ok
    return has_path_via_missing_nodes


def get_ccs_on_same_gpu(bins_to_id, gpu_id, nodes_in_bin, nodes_with_in_edges_from_different_gpu,
                        nodes_with_out_edges_to_different_gpu):
    uf = UnionFind(elements=bins_to_id[gpu_id])
    visited = set()
    open = deque(sorted(nodes_in_bin, key=lambda x: x.id))
    while open:
        x = open.popleft()
        x: Node
        for y in x.out_edges:
            if y.id not in uf:
                nodes_with_out_edges_to_different_gpu[gpu_id].add(x)
                nodes_with_in_edges_from_different_gpu[y.gpu_id].add(y)
                continue
            uf.union(x.id, y.id)
    unbroken_stages = uf.sorted_components()
    return unbroken_stages


def analyze_n_clusters(nodes: List[Node], node_weight_function, max_k=10, THRESHOLD=0, manual_choose_n_clusters=True):
    """ utility to help determine number of clusters for partition_2dbin_pack"""

    def node_to_record(node):
        return {"id": node.id, "weight": node_weight_function(node)}

    records = [node_to_record(node) for node in nodes]

    records = [node_to_record(node) for node in nodes]
    X = pd.DataFrame.from_records(data=records, index="id")
    nodes_below_thresholds = X[X["weight"] <= THRESHOLD]

    # filter below-threashold items
    nodes_above_thresholds = X[X["weight"] > THRESHOLD]

    X = pd.DataFrame.from_records(data=records, index="id")
    print(X)
    Y = X.copy()
    Y['scope'] = [node.scope for node in nodes]
    print(Y)

    sse = {}
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, max_iter=1000, copy_x=True).fit(nodes_above_thresholds)
        X.loc[nodes_above_thresholds.index, "cluster"] = kmeans.labels_

        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show(block=False)

    if manual_choose_n_clusters:
        n_clusters = input("-I- choose desired number of clusters to continue...")
        n_clusters = int(n_clusters)
        return n_clusters


def partition_2dbin_pack(graph: Graph,
                         num_gpus: int,
                         n_clusters: int,
                         node_weight_function: Optional[NodeWeightFunction] = None,
                         # edge_weight_function: Optional[EdgeWeightFunction] = None,
                         use_layers_graph: bool = True,
                         THRESHOLD=0,
                         second_and_on_cluster_policy: SecondAndOnClusterPolicy = SecondAndOnClusterPolicy.BestFitBinPacking,
                         reminder_policy: ReminderPolicy = ReminderPolicy.ToLast,
                         display_cluster_sse_plot=False,
                         **kwargs
                         ):
    # Policies control whether we Actually do the bin pack (first fit) with the splits (triples).
    #    Starting from 2nd cluster it matters.
    #    However, it can cause weird communication pattern.
    # print(f"-I- THRESHOLD={THRESHOLD}")

    # Convert
    if isinstance(second_and_on_cluster_policy, type(next(iter(ReminderPolicy._value2member_map_.keys())))):
        second_and_on_cluster_policy = SecondAndOnClusterPolicy._value2member_map_[second_and_on_cluster_policy]

    if isinstance(reminder_policy, type(next(iter(ReminderPolicy._value2member_map_.keys())))):
        reminder_policy = ReminderPolicy._value2member_map_[reminder_policy]

    print(f"use_layers_graph={use_layers_graph}")
    graph.topo_sort()

    if use_layers_graph:
        work_graph, lookup = graph.layers_graph()
    else:
        work_graph, lookup = graph, None

    nodes = [n for n in work_graph.nodes if n not in work_graph.inputs]

    K = num_gpus
    if "analyze_n_clusters" in kwargs and kwargs["analyze_n_clusters"]:
        n_clusters = analyze_n_clusters(nodes, node_weight_function, max_k=10, THRESHOLD=THRESHOLD,
                                        manual_choose_n_clusters=True)
        print(f"-I- Will use n_clusters={n_clusters}")
    elif display_cluster_sse_plot:
        print("-V- displaying info about n_clusters")
        analyze_n_clusters(nodes, node_weight_function, max_k=10, THRESHOLD=THRESHOLD, manual_choose_n_clusters=False)

    # import sys
    # sys.exit(0)
    id_to_node = {node.id: node for node in nodes}
    C = n_clusters

    clusters, to_unify = make_clusters(work_graph, nodes, node_weight_function, C=C, THRESHOLD=THRESHOLD)
    bins = best_Fit_cluster(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C,
                            second_and_on_cluster_policy=second_and_on_cluster_policy,
                            reminder_policy=reminder_policy)
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

    stage_to_gpu_map = convert_handle_missing_print(bins, graph)

    return graph, stage_to_gpu_map


def convert_handle_missing_print(bins, graph, verbose=False):
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
    stage_to_gpu_map = handle_missing_stages(bins, graph, node_to_stage_map, stage_to_gpu_map)
    stage_to_nodes_map = defaultdict(list)
    for i, v in node_to_stage_map.items():
        stage_to_nodes_map[v].append(i)

    print("stage_to_gpu_map:")
    pprint(stage_to_gpu_map)
    if verbose:
        print("node_to_stage_map:")
        pprint(node_to_stage_map)
        print("stage_to_nodes_map:")
        pprint(stage_to_nodes_map)

    return stage_to_gpu_map


def handle_missing_stages(bins, graph, node_to_stage_map, stage_to_gpu_map):
    to_check = sorted(stage_to_gpu_map.keys())
    if to_check[0] != 0 or to_check[-1] != len(to_check) - 1:
        print(f"-V- stages gone, stages_ids_before: {to_check} reassigning...")

        stage_to_fixed = {prev_s: i for i, prev_s in enumerate(to_check)}

        # 1
        for n, prev_s in list(node_to_stage_map.items()):
            if prev_s in stage_to_fixed:
                fix = stage_to_fixed[prev_s]
                node_to_stage_map[n] = fix

        # 2
        for n in graph.nodes:
            if n.stage_id in stage_to_fixed:
                n.stage_id = stage_to_fixed[n.stage_id]

        # 3
        stage_to_gpu_map = defaultdict(set)
        for gpu_id, bin_nodes in bins.items():
            for n in bin_nodes:
                stage_to_gpu_map[n.stage_id].add(gpu_id)
        stage_to_gpu_map = {i: sorted(v) for i, v in stage_to_gpu_map.items()}
    return stage_to_gpu_map


if __name__ == '__main__':
    from autopipe.autopipe import build_profiled_graph
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
    graph = build_profiled_graph(model, model_args=(inputs,), n_iter=50)

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


# for gpu_id in [0,1,2,3]:
#     nn = {n for n in work_graph.nodes if n.gpu_id == gpu_id}
#     stages = sorted({n.stage_id for n in nn})
#     for s in stages:
#         nnns = sorted({n.id for n in nn if n.stage_id == s})
#         print(s, nnns)
