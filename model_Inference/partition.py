import numpy as np


def partition(graph: np.ndarray, vertixWeights: np.ndarray, nparts: int):
    open_nodes = [0]
    close_nodes = []
    parts = []
    current_part_idx = 1
    current_part = []
    current_sum = 0
    avg = vertixWeights.mean()
    max_elem = vertixWeights.max()

    while True:
        while open_nodes:
            idx = open_nodes.pop(0)
            close_nodes.append(idx)

            if current_sum + vertixWeights[idx] < avg+2*max_elem or current_part_idx == nparts:
                current_part.append(idx)
                current_sum += vertixWeights[idx]
            elif current_sum + vertixWeights[idx] > avg and current_part_idx < nparts:
                parts.append(current_part)
                current_part = []
                current_sum = 0
                open_nodes = [idx]+open_nodes
                close_nodes.remove(idx)
                current_part_idx += 1
                continue

            for i in graph[idx]:
                if i not in close_nodes and i not in open_nodes:
                    open_nodes.append(i)

        if len(close_nodes) == graph.shape[0]:
            break

        if current_part_idx < nparts:
            parts.append(current_part)
            current_part = []
            current_sum = 0
            current_part_idx += 1

        open_nodes.append(
            (set(range(graph.shape[0])) - set(close_nodes)).pop())

    return parts+[current_part]
