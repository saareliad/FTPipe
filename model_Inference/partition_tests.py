from partition import partition
import partition_graph as pg
import numpy as np

if __name__ == "__main__":
    w = np.array([10, 8, 1, 5, 4, 2])
    nparts = 2
    g = np.array([[1], [0, 2], [1, 3, 4], [2, 5], [2, 5], [3, 4]])
    print(partition(g, w, nparts))

    mat = [[0, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 0, 1, 1, 0],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 0, 1, 1, 0]]
    mat = np.matrix(mat)
    print(pg.partition_graphs(mat, w, nparts))
