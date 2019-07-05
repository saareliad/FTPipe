from pytorch_Gpipe import visualize, visualize_with_profiler
import torch
import pytest
from pytorch_Gpipe.utils import traverse_model
from .dummy_nets import arithmeticNet, combinedTreeNet, netWithTensors, treeNet


def test_max_depth(device='cpu'):
    '''using max depth the graph will have all basic layers and input tensors'''
    depth = 5
    net = treeNet(depth).to(device)
    x = torch.randn(50, 10).to(device)
    graph = visualize(net, x)

    assert len(graph) == (2**(depth+1) + 2**(depth+1)-1 + 1)


def test_custom_depth(device='cpu'):
    '''using custom depth the graph will have all basic layers and input tensors upto that depth'''
    depth = 5
    profiled_depth = 3
    net = treeNet(depth).to(device)
    x = torch.randn(50, 10).to(device)
    graph = visualize(net, x, max_depth=profiled_depth)

    assert len(graph) == (
        2**(profiled_depth+1) + 2**(profiled_depth+1)-1 + 1)


def test_custom_basic_blocks(device='cpu'):
    ''' if a basic_block list is provided then those classes will not be broken down to their sub layers'''
    depth = 5
    x = torch.randn(50, 10, device=device)
    net = treeNet(depth).to(device)
    graph = visualize(net, x, basic_blocks=[treeNet])

    assert len(graph) == (3+1)


def test_custom_depth_and_blocks(device='cpu'):
    '''custom depth and basic_blocks can be used together in order to control what specific layers will be profiled/broken down'''
    depth = 10
    profiled_depth = 7
    net = combinedTreeNet(depth).to(device)
    x = torch.randn(50, 10, device=device)
    graph = visualize(net, x, max_depth=profiled_depth,
                      basic_blocks=[treeNet])

    assert len(graph) == (2 + profiled_depth + 1)


def test_does_not_induce_side_effects(device='cpu'):
    '''the graph construction will not have side effects on the model structure or on it's output'''
    depth = 5
    net = treeNet(depth).to(device)
    x = torch.randn(50, 10, device=device)
    expected_y = net(x)
    expected_modules = list(net.modules())
    expected_data = list(net.buffers()) + list(net.parameters())
    visualize(net, x)

    actual_modules = list(net.modules())
    actual_data = list(net.buffers()) + list(net.parameters())
    actual_y = net(x)
    assert len(expected_modules) == len(actual_modules)
    assert expected_modules == actual_modules

    assert len(expected_data) == len(actual_data)
    assert expected_data == actual_data

    torch.testing.assert_allclose(actual_y, expected_y)


def test_arithmetic_ops_are_also_profiled(device='cpu'):
    '''the graph will include also arithmetic ops'''
    net = arithmeticNet().to(device)
    x = torch.randn(50, 10, device=device)

    graph = visualize(net, x)

    assert len(graph) == 4


def test_buffers_and_parameters(device='cpu'):
    '''buffers and parameters that are not part of profiled scope will be part of the graph'''
    net = netWithTensors().to(device)
    x = torch.randn(50, 10, device=device)

    graph = visualize(net, x)

    assert len(graph) == (1+2+2+1)


def test_custom_weights(device='cpu'):
    '''a dictionary from scopes to weights can be provided to the graph'''
    net = netWithTensors().to(device)
    x = torch.randn(50, 10, device=device)

    scopes = list(map(lambda t: t[1], traverse_model(net))) + ['input0']

    custom_weights = {scope: 13 for scope in scopes}
    graph = visualize(net, x, weights=custom_weights)

    assert (len(list(n.weight == 13 for n in graph.nodes if n.scope in scopes))) > 0
    assert all(n.weight == 13 for n in graph.nodes if n.scope in scopes)


def test_weights_from_profiler(device='cpu'):
    '''the graph can use weights taken from our profiler'''
    depth = 10
    profiled_depth = 7
    net = combinedTreeNet(depth).to(device)
    x = torch.randn(50, 10, device=device)
    graph = visualize_with_profiler(
        net, x, max_depth=profiled_depth, basic_blocks=[treeNet])

    scopes = list(map(lambda t: t[1], traverse_model(
        net, depth=profiled_depth, basic_block=[treeNet])))

    scope_weights = filter(lambda n: n.scope in scopes, graph.nodes)
    scope_weights = list(map(lambda n: n.weight, scope_weights))

    assert len(scope_weights) == len(scopes)
    assert all(isinstance(w, tuple) for w in scope_weights)


def test_control_flow(device='cpu'):
    '''the graph recreates the model control flow accuratly'''
    depth = 3
    net = treeNet(depth).to(device)
    x = torch.randn(50, 10, device=device)
    graph = visualize(net, x)
    it = graph[0]
    assert it.scope == 'input0'
    it = list(it.out_nodes)[0]
    i = 1
    # this net is computed sequentially
    while len(it.out_nodes) > 0:
        if i % 4 == 1:
            assert it.scope.endswith('[left]')
        elif i % 4 == 2:
            assert it.scope.endswith('[middle]')
        elif i % 4 == 3:
            assert it.scope.endswith('[right]')
        else:
            assert it.scope.endswith('[middle]')
        i += 1
        it = list(it.out_nodes)[0]

    assert i == len(list(traverse_model(net)))


def test_output_shapes(device='cpu'):
    '''the graph models the intermidiate tensor shapes correctly'''
    depth = 5
    net = combinedTreeNet(depth).to(device)
    x = torch.randn(50, 10, device=device)
    graph = visualize(net, x)

    actual = map(lambda n: n.outputs, graph.nodes)
    actual = list(map(lambda out: list(out)[0], actual))

    expected = [(50, 10) for _ in graph.nodes]

    for e, a in zip(expected, actual):
        assert e == a


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_works_with_cuda():
    '''test that the cuda version also works'''
    test_max_depth(device='cuda')

    test_custom_depth(device='cuda')

    test_custom_basic_blocks(device='cuda')

    test_custom_depth_and_blocks(device='cuda')

    test_does_not_induce_side_effects(device='cuda')

    test_arithmetic_ops_are_also_profiled(device='cuda')

    test_buffers_and_parameters(device='cuda')

    test_custom_weights(device='cuda')

    test_weights_from_profiler(device='cuda')
