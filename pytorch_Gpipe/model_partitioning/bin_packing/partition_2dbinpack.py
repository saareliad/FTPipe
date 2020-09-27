import warnings
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import count
from pprint import pprint
from typing import Optional, List, Dict, Any, Union, Set

import matplotlib.pyplot as plt
import networkx as nx
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

class ReminderPolicy(Enum):
    ToLast = "last"
    ToMin = "min"


class SecondAndOnClusterPolicy(Enum):
    FirstFitBinPacking = "first_fit"
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
                zz = cluster_D[z]
                if isinstance(zz, list):
                    v.extend(zz)
                else:
                    v.append(zz)
                v.sort(key=lambda y: y.Index)
                cluster_D[x] = v

            # delete what we unified.
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
            pass

        N = n_i // K

        cluster_for_split = cluster if not reminder else cluster[:-reminder]

        split = list(maketree(n=N, iterable=cluster_for_split))

        if reminder > 0:
            if reminder_policy == ReminderPolicy.ToLast:
                warnings.warn(
                    f"cluster {c_i} is problematic {c_i}, {n_i}%{K}!=0, will put reminding {reminder} nodes in last partition")
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

        # TODO: this can all happen normally with graph matirx representation and mamtuls
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

    # fix to 8->9 and 8->10 and 9->10 prolem which is reduced to 8->9->10 and useless data jump between gpus
    # TODO
    # TODO
    # TODO
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO  topo sort out edges for nodes.
    # TODO: topo sort all node ids in the graph
    # TODO: go over candidates by distance (Can use matrices representation...)

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


def first_fit_cluster(K: int, clusters, id_to_node: Dict[int, Node],
                      to_unify: Dict[int, List[Union[List, Any]]], C: int,
                      second_and_on_cluster_policy: SecondAndOnClusterPolicy = SecondAndOnClusterPolicy.FirstFitBinPacking,
                      reminder_policy: ReminderPolicy = ReminderPolicy.ToLast,
                      ):
    # result
    bins = defaultdict(list)
    bin_weights = heapdict({i: 0 for i in range(K)})
    # get splits
    all_splits = get_all_splits(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C,
                                reminder_policy=reminder_policy)

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


def stages_from_bins(graph: Graph, bins: Dict[int, List[Node]], id_to_node_worked_on: Dict[int, Node]):
    stage_id_generator = count()

    # shallow copy bins:
    bins_to_id = {i: set(n.id for n in v) for i, v in bins.items()}

    nodes_with_out_edges_to_different_gpu = defaultdict(set)
    nodes_with_in_edges_from_different_gpu = defaultdict(set)
    for gpu_id, v in bins.items():
        # Find all connected components in the bin
        uf = UnionFind(elements=bins_to_id[gpu_id])
        visited = set()
        open = deque(sorted(v, key=lambda x: x.id))
        while open:
            x = open.popleft()
            x: Node
            for y in x.out_edges:
                if y.id not in uf:
                    nodes_with_out_edges_to_different_gpu[gpu_id].add(x)
                    nodes_with_in_edges_from_different_gpu[y.gpu_id].add(y)
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
                        # TODO: missing_topo_sort_id in graph is redundant, but here just in case
                        if missing_topo_sort_id in graph and missing_topo_sort_id in id_to_node_worked_on:
                            # TODO: this is too coarse grained. Need to actually check if there is a cycle.
                            try:
                                cur_nodes = [id_to_node_worked_on[x] for x in cur_set]
                            except KeyError as e:
                                print("-V- Known bug/issue (currently happens in METIS only?). Raising extra info")
                                print("-V- cur_nodes = [id_to_node_worked_on[x] for x in cur_set]")
                                print("-V- id_to_node_worked_on:", id_to_node_worked_on)
                                print("-V- cur_set", cur_set)
                                print("-V- Finding problematic keys:")
                                _first = True
                                for x in cur_set:
                                    if x not in id_to_node_worked_on:
                                        if _first:
                                            print("-V- First problematic key (node_id):", x)
                                            _first = False
                                        else:
                                            print("-V- Problematic key (node_id):", x)
                                raise e

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
                    if not is_ok:
                        broken_stages_for_unbroken_stage.append(cur_set)
                        cur_set = list()

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

        broken_stages: List[List[int]]

        # Give a dummy stage id
        for dummy_stage_id, broken_stage in zip(stage_id_generator, broken_stages):
            # Try to break stage if not TOPOLOGICALLY sorted.
            for n in broken_stage:
                graph[n].stage_id = dummy_stage_id

        # Unify redundant stages
        # cannonize_partition_indices(graph)

        # Current problem:
        # that we have 7->8>9 and we can spare 8,
        # however 8 is in the same GPU (bin) with 10 somehow
        # ids 34,35 -> 41
        # TODO: this should be fix with topo sort edges so I am about to deprecate this fix as its ofter unessesary,
        # And there is a bug in the way we do it (we look at stages we did not break yet. as its inside a for)

        EXPERIMENTAL_FIX_ONE_TO_ONE = False
        if EXPERIMENTAL_FIX_ONE_TO_ONE:
            fixes = []

            @dataclass
            class Fix:
                stage_id: int
                dst_stage_id: int
                gpu_id: int
                dst_gpu_id: int

            for broken_stage in broken_stages:
                # if not broken_stage:
                #     continue
                #
                tmp = graph[broken_stage[0]]
                gpu_id = tmp.gpu_id
                stage_id = tmp.stage_id
                all_outputs = set()
                all_gpus_ids = set()
                for n in broken_stage:
                    n = graph[n]
                    for out in n.out_edges:
                        if out.stage_id != n.stage_id:
                            all_outputs.add(out.stage_id)
                            if len(all_outputs) > 1:
                                break
                            if len(all_outputs) == 1 and len(all_gpus_ids) == 1 and out.gpu_id not in all_gpus_ids:
                                print("all_gpus_ids, out.gpu_id, out.id, out.scope:")
                                print(all_gpus_ids, out.gpu_id, out.id, out.scope, )

                                print("all graph:")
                                print("{}  |  {}  |  {} ".format("id", "stage", "gpu"))

                                for x in graph.nodes:
                                    print("{}  |  {}  |  {} ".format(x.id, x.stage_id, x.gpu_id))
                                raise ValueError(f"Detected problematic GPU id {out.gpu_id}")
                            all_gpus_ids.add(out.gpu_id)

                    if len(all_outputs) > 1:
                        break
                if len(all_outputs) == 1:
                    dst_stage_id = next(iter(all_outputs))
                    assert len(all_gpus_ids) == 1
                    dst_gpu_id = next(iter(all_gpus_ids))

                    # Important:
                    # Unify only if in same GPU.
                    print(f"-I- one2one stage: {stage_id}->{dst_stage_id}  ||| GPU: {gpu_id}->{dst_gpu_id}")
                    if dst_gpu_id == gpu_id:
                        fixes.append(
                            Fix(stage_id=stage_id, gpu_id=gpu_id, dst_gpu_id=dst_gpu_id, dst_stage_id=dst_stage_id))

            print(f"-I- got {len(fixes)} fixes.")
            d_stage_to_dst = dict()
            d_gpu_to_dst = dict()
            for fix in fixes:
                d_stage_to_dst[fix.stage_id] = fix.dst_stage_id
                d_gpu_to_dst[fix.gpu_id] = fix.dst_gpu_id

            def last_dst_stage(stage_id, stack=None):
                if stage_id in d_stage_to_dst:
                    if stack is None:
                        stack = []
                    stack.append(stage_id)
                    return last_dst_stage(d_stage_to_dst[stage_id], stack)
                else:
                    for v in stack:
                        d_stage_to_dst[v] = stage_id
                    return stage_id

            # Apply on all to fix the d_stage_to_dst dict to last
            for stage_id in list(d_stage_to_dst.keys()):
                last_dst_stage(stage_id)

            print(f"-I- fixing d_stage_to_dst: {d_stage_to_dst}")
            for i, v in d_stage_to_dst.items():
                a = [n.id for n in graph.nodes if n.stage_id == i]
                b = [n.id for n in graph.nodes if n.stage_id == v]
                print("-I-merge:", i, v, a, b)

            # Replace
            for n in graph.nodes:
                if n.stage_id in d_stage_to_dst:
                    dst = d_stage_to_dst[n.stage_id]
                    n.stage_id = dst

            #     # NOTE: this create an empty stage, but it will be wiped out by following functions

    # cannonize_partition_indices(graph) <--- TODO: redundant
    # break cycles <--- TODO: redundant


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
                         second_and_on_cluster_policy: SecondAndOnClusterPolicy = SecondAndOnClusterPolicy.FirstFitBinPacking,
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
    bins = first_fit_cluster(K, clusters, id_to_node=id_to_node, to_unify=to_unify, C=C,
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


def convert_handle_missing_print(bins, graph):
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
