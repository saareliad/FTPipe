
import itertools


import networkx as nx

import nxmetis
from nxmetis import exceptions
from nxmetis import metis
from nxmetis import types
import pytest


def make_cycle(n):
    xadj = list(range(0, 2 * n + 1, 2))
    adjncy = list(
        itertools.chain.from_iterable(
            zip(itertools.chain([n - 1], range(n - 1)),
                itertools.chain(range(1, n), [0]))))
    return xadj, adjncy


def assert_equal(a, b):
    assert a == b


def assert_not_equal(a, b):
    assert not (a == b)


class TestMetis(object):

    def setup_method(self, test_method):
        self.node_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                          1, 2, 3, 4, 5, 6]
        self.G = nx.Graph()
        nx.add_path(self.G, self.node_list)
        self.G.add_edge(self.node_list[-1], self.node_list[0])

    def test_node_nested_dissection_unweighted(self):

        node_ordering = nxmetis.node_nested_dissection(self.G)
        assert_equal(len(self.G), len(node_ordering))
        assert_equal(set(self.G), set(node_ordering))

        # Tests for exercising package's ability to handle self-loops
        # METIS crashes on self loops. networkx-metis should not
        self.G.add_edge(1, 1)
        self.G.add_edge('a', 'a')
        node_ordering = nxmetis.node_nested_dissection(self.G)
        assert_equal(len(self.G), len(node_ordering))
        assert_equal(set(self.G), set(node_ordering))

    def test_partition(self):
        partition = nxmetis.partition(self.G, 4)
        # When we choose one node from one part of the partitioned Graph,
        # It must be adjacent to one or more of the nodes in the same part.
        # This is to verify the continuity of the chain of nodes.
        parts = partition[1]  # List containing partitioned node lists

        assert_equal(partition[0], 4)
        assert_equal(len(partition[1]), 4)

        for part in parts:
            assert_not_equal(0, len(part))  # Non-empty set
            assert_equal(
                len(part), len(set(part)))  # Duplicate-free
            assert (nx.is_connected(self.G.subgraph(part)))  # Connected

        # Disjoint sets
        for part1, part2 in itertools.combinations(parts, 2):
            assert_equal(set(), set(part1) & set(part2))

        # These parts must be exhaustive with the node list of the Graph
        parts_combined = parts[0] + parts[1] + parts[2] + parts[3]
        assert_equal(set(parts_combined), set(self.G))

    def test_vertex_separator(self):
        sep, part1, part2 = nxmetis.vertex_separator(self.G)

        # The two separator nodes must not be present in the
        # two bisected chains
        assert (sep[0] not in part1)
        assert (sep[0] not in part2)
        assert (sep[1] not in part1)
        assert (sep[1] not in part2)

        # There should be two different separator nodes
        assert_equal(len(sep), 2)
        assert_not_equal(sep[0], sep[1])

        # The lists should be exhaustive with the node list of the Graph
        assert_equal(set(sep) | set(part1) | set(part2),
                     set(self.G))

        # The parts must be disjoint sets
        assert_equal(set(), set(part1) & set(part2))

        # Non-empty set
        assert_not_equal(len(part1), 0)
        assert_not_equal(len(part2), 0)

        # Duplicate-free
        assert_equal(len(part1), len(set(part1)))
        assert_equal(len(part2), len(set(part2)))

        # Connected
        assert (nx.is_connected(self.G.subgraph(part1)))
        assert (nx.is_connected(self.G.subgraph(part2)))

    # def test_MetisOptions(self):
    #     n = 16
    #     xadj, adjncy = make_cycle(n)
    #     options = types.MetisOptions(niter=-2)
    #     nose.tools.assert_raises_regexp(exceptions.MetisError,
    #                                     'Input Error: Incorrect niter.',
    #                                     metis.part_graph, xadj, adjncy, 2,
    #                                     options=options)
