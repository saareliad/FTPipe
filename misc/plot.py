import matplotlib.pyplot as plt
import itertools
import numpy as np
from typing import NamedTuple, Union


def plot_fit(fit_res: Union[NamedTuple, dict], fig=None, log_loss=False, legend=None, loss_per_batch=False):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :return: The figure.
    """
    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10),
                                 sharex='col', sharey=False)
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(['train', 'test'], ['loss', 'acc'])
    for idx, (traintest, lossacc) in enumerate(p):
        ax = axes[idx]
        attr = f'{traintest}_{lossacc}'
        if isinstance(fit_res, NamedTuple):
            data = getattr(fit_res, attr)
        else:
            data = fit_res[attr]
        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        ax.set_title(attr)
        loss_name = "Iteration" if loss_per_batch else "Epoch"
        if lossacc == 'loss':
            ax.set_xlabel(f'{loss_name} #')
            ax.set_ylabel('Loss')
            if log_loss:
                ax.set_yscale('log')
                ax.set_ylabel('Loss (log)')
        else:
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Accuracy (%)')
        if legend:
            ax.legend()

    return fig, axes


def plot_grad_norm(fit_res: Union[NamedTuple, dict], fig=None, legend=None, **kw):
    total_norms = sum("grad_norm" in key for key in fit_res.keys())
    assert(total_norms % 2 == 0)  # TODO support for un-even...
    # total_norms / 2
    if fig is None:
        fig, axes = plt.subplots(nrows=1+ total_norms//2, ncols=2, figsize=(16, 10),
                                 sharex='col', sharey=False)
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    all_norms = [key for key in fit_res.keys() if "grad_norm" in key]
    p = all_norms + ["train_acc", "test_acc"]
    for idx, attr in enumerate(p):
        ax = axes[idx]
        if isinstance(fit_res, NamedTuple):
            data = getattr(fit_res, attr)
        else:
            data = fit_res[attr]

        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        # bwd compatability...
        attr = "last_partition_grad_norm" if attr == "total_grad_norm" else attr
        ax.set_title(attr)
        ax.set_xlabel('Epoch #')

        if "acc" in attr:
            ax.set_ylabel('Accuracy (%)')
        else:
            ax.set_ylabel('Norm')

        if legend:
            ax.legend()

    return fig, axes



def plot_gap(fit_res: Union[NamedTuple, dict], fig=None, legend=None, **kw):
    total = sum("gap" in key for key in fit_res.keys())
    assert(total % 2 == 0)  # TODO support for un-even...
    # total_norms / 2
    if fig is None:
        fig, axes = plt.subplots(nrows=1+ total//2, ncols=2, figsize=(16, 10),
                                 sharex='col', sharey=False)
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    all_ = [key for key in fit_res.keys() if "gap" in key]
    max_len = max(len(key) for key in all_)
    for key in all_:
        if len(key) == 0:
            key += [0]*max_len

    p = all_ + ["train_acc", "test_acc"]
    for idx, attr in enumerate(p):
        ax = axes[idx]
        if isinstance(fit_res, NamedTuple):
            data = getattr(fit_res, attr)
        else:
            data = fit_res[attr]

        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        # bwd compatability...
        # attr = "last_partition_grad_norm" if attr == "total_grad_norm" else attr
        ax.set_title(attr)
        ax.set_xlabel('Epoch #')

        if "acc" in attr:
            ax.set_ylabel('Accuracy (%)')
        else:
            ax.set_ylabel('Gap [sum of L2 norms]')

        if legend:
            ax.legend()

    return fig, axes

