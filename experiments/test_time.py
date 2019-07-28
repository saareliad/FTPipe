
import timeit
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn, optim as optim

# from .run_test import num_classes, num_batches, batch_size, image_w, image_h
num_classes = 1000
num_batches = 3
batch_size = 120
image_w = 224
image_h = 224


def test_resnet50_time():
    num_repeat = 10

    stmt = "train(model)"

    setup = "from .model_parallel_resnet50 import make_pipeline_resnet\n" \
            "model = make_pipeline_resnet(20)"
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
        print(
            f'data parallel has speedup of {(rn_mean / dp_mean - 1) * 100} relative to single gpu')
        plot([mp_mean, rn_mean, dp_mean],
             [mp_std, rn_std, dp_std],
             ['Model Parallel', 'Single GPU', 'Data Parallel'],
             'mp_vs_rn_vs_dp.png')

    print(
        f'pipeline has speedup of {(rn_mean / mp_mean - 1) * 100} relative to single gpu')
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

    one_hot_indices = torch.LongTensor(batch_size).random_(
        0, num_classes).view(batch_size, 1)

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for b in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes).scatter_(
            1, one_hot_indices, 1)

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
