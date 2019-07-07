import pytest
from pytorch_Gpipe.pipeline import *
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
import matplotlib.pyplot as plt
import numpy as np
import timeit

num_classes = 1000
num_batches = 3
batch_size = 120
image_w = 224
image_h = 224

plt.switch_backend('Agg')


class PrintLayer(nn.Module):
    def __init__(self, to_print: str = None):
        super(PrintLayer, self).__init__()
        self.to_print = to_print

    def forward(self, input):
        if self.to_print is None:
            to_print = f'shape is {input.shape}'
        else:
            to_print = self.to_print

        print(to_print)
        return input


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        dev1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.counter = CycleCounter(2)
        self.act_saving = ActivationSavingLayer(dev1, counter=self.counter)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to(dev1)

        self.seq1 = LayerWrapper(self.seq1, 0, dev1, output_shapes=(
            (1, 512, 28, 28),), counter=self.counter)

        dev2 = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        ).to(dev2)

        self.seq2 = SyncWrapper(self.seq2, dev2, 1, output_shapes=(
            (1, 2048, 1, 1),), counter=self.counter)

        self.fc.to(dev2)
        self.fc = LayerWrapper(self.fc, 1, dev2, output_shapes=(
            (1, 1000),), counter=self.counter)

    def forward(self, x):
        x = self.act_saving(x)
        x = self.seq1(x)
        x = self.seq2(x)
        return self.fc(x.view(x.size(0), -1))


def make_pipeline_resnet(microbatch_size: int, num_classes=num_classes):
    inner_module = ModelParallelResNet50(num_classes=num_classes)
    counter = inner_module.counter
    wrappers = [inner_module.act_saving, inner_module.seq2]
    input_shape = (1, 3, 224, 224)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    return PipelineParallel(inner_module, microbatch_size, 2, input_shape, counter=counter, wrappers=wrappers,
                            main_device=device)


def test_split_size():
    num_repeat = 10

    stmt = "train(model)"

    means = []
    stds = []
    split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]

    for split_size in split_sizes:
        setup = "model = make_pipeline_resnet(%d)" % split_size
        pp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        means.append(np.mean(pp_run_times))
        stds.append(np.std(pp_run_times))

    fig, ax = plt.subplots()
    ax.plot(split_sizes, means)
    ax.errorbar(split_sizes, means, yerr=stds, ecolor='red', fmt='ro')
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xlabel('Pipeline Split Size')
    ax.set_xticks(split_sizes)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig("split_size_tradeoff.png")
    plt.close(fig)
