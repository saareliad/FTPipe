from copy import deepcopy
from itertools import chain

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_Gpipe.pipeline import DelayedBatchNorm

NUM_MICRO_BATCHES = 4

# taken from torchGpipe repo

# TODO need to test after implementation


def tilt_dist(x):
    # Tilt variance by channel.
    rgb = x.transpose(0, 1)
    rgb[0] *= 1
    rgb[1] *= 10
    rgb[2] *= 100

    # Tilt mean by single batch.
    for i, single in enumerate(x):
        single += 2**i

    return x


def chunked_forward(model, x, micro_batches=NUM_MICRO_BATCHES):
    output_micro_batches = []

    for chunk in x.chunk(micro_batches):
        output_micro_batches.append(model(chunk))

    return torch.cat(output_micro_batches)


@pytest.mark.parametrize('micro_batches', [1, 4])
@pytest.mark.parametrize('x_requires_grad', [True, False])
def test_transparency(micro_batches, x_requires_grad):
    bn = nn.BatchNorm2d(3)
    dbn = DelayedBatchNorm.convert(
        deepcopy(bn), num_micro_batches=micro_batches)

    x1 = torch.rand(16, 3, 224, 224)
    x1 = tilt_dist(x1)
    x2 = x1.clone()
    x1.requires_grad = x_requires_grad
    x2.requires_grad = x_requires_grad

    output1 = chunked_forward(bn, x1, micro_batches=micro_batches)
    output2 = chunked_forward(dbn, x2, micro_batches=micro_batches)

    assert torch.allclose(output1, output2, atol=1e-4)

    output1.mean().backward()
    output2.mean().backward()

    assert torch.allclose(bn.weight.grad, dbn.weight.grad, atol=1e-4)

    if x_requires_grad:
        assert x1.grad is not None
        assert x2.grad is not None
        assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


@pytest.mark.parametrize('momentum', [0.1, None])
def test_running_stats(momentum):
    bn = nn.BatchNorm2d(3, momentum=momentum)
    dbn = DelayedBatchNorm.convert(
        deepcopy(bn), num_micro_batches=NUM_MICRO_BATCHES)

    x = torch.rand(16, 3, 224, 224)
    x = tilt_dist(x)

    bn(x)
    chunked_forward(dbn, x)

    assert torch.allclose(bn.running_mean, dbn.running_mean, atol=1e-4)
    assert torch.allclose(bn.running_var, dbn.running_var, atol=1e-4)


def test_convert():
    bn = nn.BatchNorm2d(3, track_running_stats=False)
    bn = DelayedBatchNorm.convert(bn, num_micro_batches=NUM_MICRO_BATCHES)
    assert type(bn) is nn.BatchNorm2d  # because of track_running_stats=False

    dbn = DelayedBatchNorm(3, num_micro_batches=NUM_MICRO_BATCHES)
    dbn_again = DelayedBatchNorm.convert(
        dbn, num_micro_batches=NUM_MICRO_BATCHES)
    assert dbn.weight is dbn_again.weight
    assert dbn.bias is dbn_again.bias
    assert dbn.running_mean is dbn_again.running_mean
    assert dbn.running_var is dbn_again.running_var


def test_eval():
    bn = nn.BatchNorm2d(3)
    dbn = DelayedBatchNorm.convert(
        deepcopy(bn), num_micro_batches=NUM_MICRO_BATCHES)

    x = torch.rand(16, 3, 224, 224)
    x = tilt_dist(x)

    bn(x)
    chunked_forward(dbn, x)

    bn.eval()
    dbn.eval()

    assert torch.allclose(bn(x), dbn(x), atol=1e-4)


def test_optimize():
    bn = nn.BatchNorm2d(3)
    dbn = DelayedBatchNorm.convert(
        deepcopy(bn), num_micro_batches=NUM_MICRO_BATCHES)

    opt = optim.SGD(chain(bn.parameters(), dbn.parameters()), lr=1.0)

    for i in range(5):
        x = torch.rand(16, 3, 224, 224)
        x = tilt_dist(x)

        # train
        y = bn(x)
        a = y.sum()
        a.backward()

        y = chunked_forward(dbn, x)
        b = y.sum()
        b.backward()

        opt.step()

        # eval
        bn.eval()
        dbn.eval()

        with torch.no_grad():
            assert torch.allclose(bn(x), dbn(x), atol=1e-1 * (10**i))


def test_conv_bn():
    bn = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
    dbn = DelayedBatchNorm.convert(
        deepcopy(bn), num_micro_batches=NUM_MICRO_BATCHES)

    x = torch.rand(16, 3, 224, 224)
    x = tilt_dist(x)

    opt = optim.SGD(chain(bn.parameters(), dbn.parameters()), lr=0.1)

    # 1st step
    a = bn(x)
    b = chunked_forward(dbn, x)

    # Outputs are different. (per-mini-batch vs. per-micro-batch)
    assert not torch.allclose(a, b)

    a.sum().backward()
    b.sum().backward()
    opt.step()
    opt.zero_grad()

    # Conv layers are also trained differently because of their different outputs.
    assert not torch.allclose(bn[0].weight, dbn[0].weight)

    # But BNs track identical running stats.
    assert torch.allclose(bn[1].running_mean, dbn[1].running_mean, atol=1e-4)
    assert torch.allclose(bn[1].running_var, dbn[1].running_var, atol=1e+3)

    # 2nd step
    a = bn(x)
    b = chunked_forward(dbn, x)
    a.sum().backward()
    b.sum().backward()

    # BNs can't track identical running stats due to the different conv layers.
    assert not torch.allclose(
        bn[1].running_mean, dbn[1].running_mean, atol=1e-4)
    assert not torch.allclose(bn[1].running_var, dbn[1].running_var, atol=1e+3)


def test_x_requiring_grad():
    dbn = DelayedBatchNorm(3, num_micro_batches=NUM_MICRO_BATCHES)

    x = torch.rand(16, 3, 224, 224, requires_grad=True)
    x = tilt_dist(x)

    chunked_forward(dbn, x)

    assert not dbn.sum.requires_grad
    assert dbn.sum.grad_fn is None
