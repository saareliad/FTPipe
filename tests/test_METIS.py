import pytest

from pytorch_Gpipe.METIS import METIS_partition, mdbglvl_et


# -------------------------------------------------------------------------
# basic tests
# -------------------------------------------------------------------------


def example_adjlist():
    return [[1, 2, 3, 4], [0], [0], [0], [0, 5], [4, 6], [13, 5, 7],
            [8, 6], [9, 10, 11, 12, 7], [8], [8], [8], [8], [14, 6], [13, 15],
            [16, 17, 18, 14], [15], [15], [15]]


def test_1():
    adjlist = example_adjlist()

    print("Testing k-way cut")
    cuts, parts = METIS_partition(adjlist, 3, algorithm="metis",
                                  dbglvl=mdbglvl_et.METIS_DBG_ALL)
    assert cuts == 2
    assert set(parts) == set([0, 1, 2])

    print("Testing recursive cut")
    cuts, parts = METIS_partition(adjlist, 3, algorithm="metis_recursive",
                                  dbglvl=mdbglvl_et.METIS_DBG_ALL)
    assert cuts == 2
    assert set(parts) == set([0, 1, 2])

    # print("METIS appears to be working.")


def test_2():
    nVertices = 6
    nParts = 2

    # Indexes of starting points in adjacent array
    adj_idx = [0, 2, 5, 7, 9, 12, 14]

    # Adjacent vertices in consecutive index order
    adjv = [1, 3, 0, 4, 2, 1, 5, 0, 4, 3, 1, 5, 4, 2]

    adjlist = [adjv[adj_idx[i]:adj_idx[i+1]] for i in range(nVertices)]

    # Weights of vertices
    # if all weights are equal then can be set to NULL
    nodew = [i*nVertices for i in range(nVertices)]

    # int ret = METIS_PartGraphRecursive( & nVertices, & nWeights, xadj, adjncy,
    #            NULL, NULL, NULL, & nParts, NULL,
    #            NULL, NULL, & objval, part)

    cuts, parts = METIS_partition(
        adjlist, nParts, algorithm="metis", nodew=nodew, contig=1)

    assert len(set(parts)) == nParts
