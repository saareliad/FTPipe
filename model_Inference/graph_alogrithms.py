# This class represents an directed graph
# using adjacency list representation
from collections import defaultdict


def strongly_connected_components_iterative(vertices, edges):
    """
    This is a non-recursive version of strongly_connected_components_path.

    Examples
    --------
    Example from Gabow's paper [1]_.

    >>> vertices = [1, 2, 3, 4, 5, 6]
    >>> edges = {1: [2, 3], 2: [3, 4], 3: [], 4: [3, 5], 5: [2, 6], 6: [3, 4]}
    >>> for scc in strongly_connected_components_iterative(vertices, edges):
    ...     print(scc)
    ...
    set([3])
    set([2, 4, 5, 6])
    set([1])
    """
    identified = set()
    stack = []
    index = {}
    boundaries = []

    for v in vertices:
        if v not in index:
            to_do = [('VISIT', v)]
            while to_do:
                operation_type, v = to_do.pop()
                if operation_type == 'VISIT':
                    index[v] = len(stack)
                    stack.append(v)
                    boundaries.append(index[v])
                    to_do.append(('POSTVISIT', v))
                    # We reverse to keep the search order identical to that of
                    # the recursive code;  the reversal is not necessary for
                    # correctness, and can be omitted.
                    to_do.extend(
                        reversed([('VISITEDGE', w) for w in edges[v]]))
                elif operation_type == 'VISITEDGE':
                    if v not in index:
                        to_do.append(('VISIT', v))
                    elif v not in identified:
                        while index[v] < boundaries[-1]:
                            boundaries.pop()
                else:
                    # operation_type == 'POSTVISIT'
                    if boundaries[-1] == index[v]:
                        boundaries.pop()
                        scc = set(stack[index[v]:])
                        del stack[index[v]:]
                        identified.update(scc)
                        yield scc
