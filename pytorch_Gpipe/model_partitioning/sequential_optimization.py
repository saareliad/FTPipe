import numpy as np

__all__ = ['sequential_partition']


def sequential_partition(seq, nparts):
    max_elem = max(seq)
    min_elem = min(seq)

    # just for sanity should not happen
    if max_elem == 0:
        max_elem = 1e-5

    # the algorithem is proved for elements in range [0,1]
    # perform min-max normalization
    normalized = [(e-min_elem)/(max_elem-min_elem) for e in seq]

    nElem = len(seq)

    indices = [(nElem//nparts) * i for i in range(nparts)] + [nElem]

    sums = [sum(normalized[i:j]) for i, j in zip(indices, indices[1:])]

    while True:
       # 1 find largest sum
        p = np.argmax(sums)
        max_size = sums[p]

        while True:
            # 2 find min sum
            q = np.argmin(sums)
            min_size = sums[q]

            # we've found guaranteed boundry
            if max_size <= min_size + 1:
                return indices

            # 3 update boundries and sums
            if p < q:
               # move the last element from Bq−1 to Bq
                k = q - 1
                indices[q] -= 1
                moved_elem = normalized[indices[q]]
            else:
                # move the first element of Bq+1 to Bq.
                k = q + 1
                indices[k] += 1
                moved_elem = normalized[indices[k]]

            sums[q] += moved_elem
            sums[k] -= moved_elem

            # if p==k goto 1 else goto 2
            if p == k:
                break


def test_sequential_partition():
    assert sequential_partition([1, 2, 3, 4, 5, 6], 2) == [0, 4, 6]


def test_sequential_partition_zeros():
    assert sequential_partition([0, 0], 2) == [0, 1, 2]
