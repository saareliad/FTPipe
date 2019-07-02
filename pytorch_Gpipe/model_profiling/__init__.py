from .control_flow_graph import Graph, NodeTypes, graph_builder
from .optimize_graph import optimize_graph
from .network_profiler import profileNetwork

__all__ = ['visualize', 'visualize_with_profiler']


def visualize(model, *sample_batch, max_depth=1000, basic_blocks=None, weights=None):
    graph = graph_builder(
        model, *sample_batch, max_depth=max_depth, basic_block=basic_blocks, weights=weights)

    optimize_graph(graph)

    return graph


def visualize_with_profiler(model, *sample_batch, max_depth=1000, basic_blocks=None, num_iter=1):
    layers_profile = profileNetwork(model, *sample_batch, max_depth=max_depth,
                                    basic_block=basic_blocks, num_iter=num_iter)

    return visualize(model, *sample_batch, max_depth=max_depth, basic_blocks=basic_blocks, weights=layers_profile)
