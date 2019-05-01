import pytest
from pipeline.pipeline_parallel import *
from pipeline.sync_wrapper import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import ResNet, Bottleneck, resnet50
import matplotlib.pyplot as plt
import numpy as np
import timeit

plt.switch_backend('Agg')

# TODO
"""
**ADD TESTS FOR THE FOLLOWING**

under pipeline/utils:
- conveyor_gen
- prod_line

under pipeline/sub_module_wrapper.py:
- make sure that forward propagation doesn't save activations that aren't the first ones
- make sure that the activations are saved in the correct order
- make sure that stuff is calculated correctly
- make sure that backward propagation works properly

under pipeline/pipeline_parallel.py:
- check correctness of __div_to_mbs
- check that forward works and has the same output as the same model undivided
- check that the use of backward works with the hook works like the model undivided
- take a model and create 2 identical copies, train each with the same random state in the start and make sure they are
    the same in the end (should have the same parameters and outputs at every step
- make sure it works with [1, 2, 3, 4] different devices

"""

num_classes = 1000
num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.seq2 = SyncWrapper(self.seq2, 'cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x))
        return self.fc(x.view(x.size(0), -1))


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
            .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


def test_first_pipline():
    num_repeat = 10

    stmt = "train(model)"

    setup = "model = ModelParallelResNet50()"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    setup = "model = resnet50(num_classes=num_classes).to('cuda:0')"
    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

    plot([mp_mean, rn_mean],
         [mp_std, rn_std],
         ['Model Parallel', 'Single GPU'],
         'mp_vs_rn.png')

    assert mp_mean < rn_mean
    assert rn_mean / mp_mean - 1 >= 0.3
