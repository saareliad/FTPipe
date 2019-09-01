from torch import optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pytorch_Gpipe import pipe_model


def kwargs_string(*pos_strings, **kwargs):
    return ', '.join(list(pos_strings) + [f'{key}={val}' for key, val in kwargs.items()])


def call_func_stmt(func, *params, **kwargs):
    if isinstance(func, str):
        func_name = func
    else:
        func_name = func.__name__

    params_str = [str(param) for param in params]

    return f'{func_name}({kwargs_string(*params_str, **kwargs)})'


def train(model, num_classes, num_batches, batch_shape):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    batch_size = batch_shape[0]

    one_hot_indices = torch.LongTensor(batch_size).random_(
        0, num_classes).view(batch_size, 1)

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for b in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(*batch_shape)
        labels = torch.zeros(batch_size, num_classes).scatter_(
            1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to(dev))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    print('.', end='')


def plot(means, stds, labels, fig_name, fig_label):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(fig_label)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


def create_pipeline(model, batch_shape, **kwargs):
    return pipe_model(model, sample_batch=torch.randn(*batch_shape), **kwargs)
