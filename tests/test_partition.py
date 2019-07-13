import torch
from .dummy_nets import treeNet
from pytorch_Gpipe import graph_builder, partition_graph, distribute_using_custom_weights
from pytorch_Gpipe.utils import traverse_model, model_scopes
from pytorch_Gpipe.pipeline import ActivationSavingLayer, LayerWrapper, SyncWrapper
import pytest
import torch.nn as nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_every_layer_has_a_partition():
    depth = 3
    device = 'cuda:0'
    net = treeNet(depth).to(device)
    x = torch.zeros(50, 10).to(device)

    graph = graph_builder(net, x)

    for n in graph.nodes:
        n.weight = 1

    for n in graph.nodes:
        assert n.weight == 1

    before_partition = {n.part for n in graph.nodes}
    assert before_partition == {-1}

    parts, _ = partition_graph(
        graph, num_partitions=4, weighting_function=lambda w: w)

    after = {n.part for n in graph.nodes}
    assert after == {0, 1, 2, 3}


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

    _, _, _, graph = distribute_using_custom_weights(net, weights, x)

    graph_scopes = list(map(lambda n: (n.scope, n.part), graph.nodes[1:]))

    cannonized_graph_scopes = sorted(graph_scopes, key=lambda t: t[0])

    assert len(cannonized_graph_scopes) == len(cannonized_scopes_layers)

    for (_, part_idx), (_, layer) in zip(cannonized_graph_scopes, cannonized_scopes_layers):
        expected_device = torch.device(f"cuda:{part_idx}")
        assert all(b.device == expected_device for b in layer.buffers)
        assert all(p.device == expected_device for p in layer.parameters)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='more than 2 CUDA devices required')
def test_wrappers_are_otimized_if_possible():
    device = 'cuda:0'
    depth = 10
    net = treeNet(depth).to(device)
    x = torch.zeros(50, 10).to(device)
    scopes = model_scopes(net)
    weights = {scope: 10 for scope in scopes}
    distributed_model, _, _, _ = distribute_using_custom_weights(
        net, weights, x)

    def find_layers_of_class(model: nn.Module, cls):
        return filter(lambda l: isinstance(l, cls), model.modules())

    assert len(find_layers_of_class(
        distributed_model, ActivationSavingLayer)) == 1

    assert len(find_layers_of_class(distributed_model, SyncWrapper)) == 1

    assert len(find_layers_of_class(distributed_model, LayerWrapper)) <= 2
