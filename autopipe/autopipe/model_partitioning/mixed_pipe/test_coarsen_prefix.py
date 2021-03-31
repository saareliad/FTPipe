import unittest
from types import SimpleNamespace

import torch
from torch import nn

from autopipe.autopipe.model_profiling.control_flow_graph import Graph
from autopipe.autopipe.model_partitioning.heuristics import get_weight_functions
from autopipe.autopipe.api import build_profiled_graph
from autopipe.autopipe.model_partitioning.mixed_pipe.by_prefix import coarsen_prefixes
from autopipe.autopipe.union_find import UnionFind
from autopipe.partitioning_scripts.partition_scripts_utils import choose_blocks


class C1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.5, inplace=False)
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MyTestCase(unittest.TestCase):
    def test_something(self):
        depth = 1000

        model_args = (torch.randn((1, 10)),)
        model = nn.Sequential(C1(), C1(), C1())
        graph: Graph = build_profiled_graph(model, model_args=model_args, n_iter=1, max_depth=depth)
        args = SimpleNamespace(bwd_to_fwd_ratio=1, bw=12, weight_mult_factor=1e4, auto_infer_node_bwd_to_fwd_ratio=False,
                               penalize_non_tensors=False, edge_penalty=1e4)
        node_weight_function, edge_weight_function = get_weight_functions(args)
        uf = UnionFind(elements=[n.id for n in graph.non_input_nodes])
        basic_blocks = ()
        args.special_blocks = ["C1"]
        special_blocks = choose_blocks(model, args, blocks_arg_name="special_blocks")

        prev_graph, matching, graph, uf, uf2, sb_names = coarsen_prefixes(model, graph, node_weight_function, edge_weight_function, uf, basic_blocks=basic_blocks,
                         special_blocks=special_blocks, depth=depth)

        print(sb_names)
        self.assertEqual(graph.num_nodes - graph.num_inputs, 3)


if __name__ == '__main__':
    unittest.main()
