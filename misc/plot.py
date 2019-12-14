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