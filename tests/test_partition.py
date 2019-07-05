import torch
from .dummy_nets import treeNet
from pytorch_Gpipe import visualize, partition_graph, distribute_using_custom_weights
from pytorch_Gpipe.utils import traverse_model
import pytest


def test_every_layer_has_a_partition(device='cpu'):
    depth = 3
    net = treeNet(depth).to(device)
    x = torch.zeros(50, 10).to(device)

    graph = visualize(net, x)

    for n in graph.nodes:
        n.weight = 1

    for n in graph.nodes:
        assert n.weight == 1

    before_partition = {n.part for n in graph.nodes}
    assert before_partition == {-1}

    _, parts, _ = partition_graph(
        graph, num_partitions=4, weighting_function=lambda w: w)

    after = {n.part for n in graph.nodes}
    assert after == {0, 1, 2, 3}
    assert parts == list(map(lambda n: n.part, graph.nodes))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='cuda required')
def test_every_layer_is_allocated_a_device_as_specified_by_the_graph():
    device = 'cuda:0'
    depth = 5
    net = treeNet(depth).to(device)
    x = torch.zeros(50, 10).to(device)

    scopes, layers, _ = zip(*traverse_model(net))

    scopes_and_layers = zip(scopes, layers)

    cannonized_scopes_layers = sorted(scopes_and_layers, key=lambda t: t[0])

    weights = {scope: 10 for scope in scopes}

    _, graph, _ = distribute_using_custom_weights(net, weights, x)

    graph_scopes = list(map(lambda n: (n.scope, n.part), graph.nodes[1:]))

    cannonized_graph_scopes = sorted(graph_scopes, key=lambda t: t[0])

    assert len(cannonized_graph_scopes) == len(cannonized_scopes_layers)

    for (_, part_idx), (_, layer) in zip(cannonized_graph_scopes, cannonized_scopes_layers):
        expected_device = torch.device(f"cuda:{part_idx}")
        assert all(b.device == expected_device for b in layer.buffers)
        assert all(p.device == expected_device for p in layer.parameters)
