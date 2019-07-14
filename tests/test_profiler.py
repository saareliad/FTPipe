import torch
from pytorch_Gpipe import profileNetwork
import pytest
from .dummy_nets import combinedTreeNet, treeNet


def test_max_depth(device='cpu'):
    ''' profiler with default depth will profile all basic pytorch layers '''
    depth = 7
    net = treeNet(depth).to(device)
    report = profileNetwork(net, torch.randn(
        50, 10, device=device), basic_blocks=None)

    assert len(report) == (2**(depth+1) + 2**(depth+1)-1)


def test_custom_depth(device='cpu'):
    ''' profiler with custom depth will profile all layers at that depth'''
    depth = 5
    profiled_depth = 3
    net = treeNet(depth).to(device)
    report = profileNetwork(net, torch.randn(50, 10, device=device),
                            max_depth=profiled_depth, basic_blocks=None)

    assert len(report) == (2**(profiled_depth+1) + 2**(profiled_depth+1)-1)


def test_custom_basic_block(device='cpu'):
    ''' if a basic_blocks list is provided then those classes will not be broken down to their sub layers'''
    depth = 5
    net = treeNet(depth).to(device)
    report = profileNetwork(net, torch.randn(
        50, 10, device=device), basic_blocks=[treeNet])

    assert len(report) == 3


def test_using_custom_depth_and_blocks(device='cpu'):
    '''custom depth and basic_blocks can be used together in order to control what specific layers will be profiled/broken down'''
    depth = 10
    profiled_depth = 7
    net = combinedTreeNet(depth).to(device)
    report = profileNetwork(net, torch.randn(50, 10, device=device),
                            max_depth=profiled_depth, basic_blocks=[treeNet])

    assert len(report) == (2 + profiled_depth)


def test_does_not_induce_side_effects(device='cpu'):
    '''the profiling will not have side effects on the model structure or on it's output'''
    depth = 5
    net = treeNet(depth).to(device)
    x = torch.randn(50, 10, device=device)
    expected_y = net(x)
    expected_modules = list(net.modules())
    expected_data = list(net.buffers()) + list(net.parameters())
    profileNetwork(net, x)

    actual_modules = list(net.modules())
    actual_data = list(net.buffers()) + list(net.parameters())
    actual_y = net(x)
    assert len(expected_modules) == len(actual_modules)
    assert expected_modules == actual_modules

    assert len(expected_data) == len(actual_data)
    assert expected_data == actual_data

    torch.testing.assert_allclose(actual_y, expected_y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda required')
def test_works_also_with_cuda():
    '''test that the cuda version also works'''
    test_max_depth('cuda')
    test_custom_depth('cuda')
    test_custom_basic_block('cuda')
    test_using_custom_depth_and_blocks('cuda')
    test_does_not_induce_side_effects('cuda')
