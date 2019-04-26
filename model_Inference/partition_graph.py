import numpy as np


def get_part(G: np.matrix, nodes_values: np.ndarray, curr_node: int, already_partitioned: list, part_size: int,
             curr_part: list, curr_sum: int):
    if already_partitioned.count(curr_node) > 0:
        return False, curr_part, np.inf

    avg_node_val = 3
    dist = abs(nodes_values[curr_node] + curr_sum - part_size)
    if dist < avg_node_val:
        curr_part.append(curr_node)
        return True, curr_part, dist

    if nodes_values[curr_node] + curr_sum - part_size > 0:
        return False, curr_part, np.inf

    curr_part.append(curr_node)
    already_partitioned.append(curr_node)

    best_dist = dist
    best_part = curr_part

    for i in range(G.shape[0]):
        if G[curr_node, i] == 1 and i != curr_node:
            flag, part, try_dist = get_part(G, nodes_values, i, already_partitioned, part_size, curr_part,
                                            curr_sum + nodes_values[curr_node])
            if flag and try_dist < best_dist:
                best_dist = try_dist
                best_part = part
    return True, best_part, best_dist


def partition_graphs(G: np.matrix, nodes_values: np.ndarray, num_parts: int = 8):
    parts = []
    already_partitioned = []
    part_size = sum(nodes_values) / num_parts

    for i in range(G.shape[0]):
        if already_partitioned.count(i) > 0:
            continue
        new_part = []
        flag, new_part, __ = get_part(G, nodes_values, i, already_partitioned, part_size, new_part, 0)
        if flag:
            parts.append(new_part)
            already_partitioned = already_partitioned + new_part
        if len(parts) + 1 == num_parts:
            left = [item for item in range(G.shape[0]) if item not in already_partitioned]
            parts.append(left)
            break
    return parts
