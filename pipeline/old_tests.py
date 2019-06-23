from torch import optim
from pipeline.pipeline_parallel import *
from pipeline.sync_wrapper import *
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
        super(ModelParallelResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], *args, **kwargs)

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

        self.seq1 = LayerWrapper(self.seq1, 0, dev1, output_shapes=((1, 512, 28, 28),), counter=self.counter)

        dev2 = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        ).to(dev2)

        self.seq2 = SyncWrapper(self.seq2, dev2, 1, output_shapes=((1, 2048, 1, 1),), counter=self.counter)

        self.fc.to(dev2)
        self.fc = LayerWrapper(self.fc, 1, dev2, output_shapes=((1, 1000),), counter=self.counter)

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


def test_resnet50_time():
    num_repeat = 10

    stmt = "train(model)"

    setup = "model = make_pipeline_resnet(20)"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    print('finished pipeline')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    setup = "model = resnet50(num_classes=num_classes).to('cuda:0' if torch.cuda.is_available() else 'cpu')"
    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

    print('finished single gpu')

    if torch.cuda.is_available():
        stmt = "train(model, True)"
        setup = "model = nn.DataParallel(resnet50(num_classes=num_classes)).to('cuda:0' if torch.cuda.is_available() else 'cpu')"
        dp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        dp_mean, dp_std = np.mean(dp_run_times), np.std(dp_run_times)
        print(f'Data parallel mean is {dp_mean}')

    plot([mp_mean, rn_mean],
         [mp_std, rn_std],
         ['Model Parallel', 'Single GPU'],
         'mp_vs_rn.png')

    if torch.cuda.is_available():
        print(f'data parallel has speedup of {(rn_mean / dp_mean - 1) * 100} relative to single gpu')
        plot([mp_mean, rn_mean, dp_mean],
             [mp_std, rn_std, dp_std],
             ['Model Parallel', 'Single GPU', 'Data Parallel'],
             'mp_vs_rn_vs_dp.png')

    print(f'pipeline has speedup of {(rn_mean / mp_mean - 1) * 100} relative to single gpu')
    assert mp_mean < rn_mean
    # assert that the speedup is at least 30%
    assert rn_mean / mp_mean - 1 >= 0.3


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


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size).random_(0, num_classes).view(batch_size, 1)

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for b in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes).scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to(dev))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        # print("")
        # print("======================================")
        # print(f'finished batch #{b}')

    # print('finished a train run!')


if __name__ == "__main__":
    test_split_size()
    test_resnet50_time()
