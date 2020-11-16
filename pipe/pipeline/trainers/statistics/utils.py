from itertools import chain

import torch
from torch.optim import Optimizer

from pipe.pipeline.trainers.statistics import Stats


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.record = []

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get_avg(self):
        return self.sum / self.count


class AccuracyMeter(AverageMeter):
    def __init__(self):
        super().__init__()

    def update(self, val, n=1):
        """ just to supoort adding num correct instead of accuracy """
        self.sum += val
        self.count += n

    def get_avg(self):
        return (self.sum / self.count) * 100


def try_record_real_gap_from_current(statistics: Stats,
                                     optimizer: Optimizer,
                                     real_theta,
                                     pre_computed_gap=None,
                                     gap_name="gap"):
    """ calculates gap between model parameters and a given set of parameters, real_theta
        real_theta: Given set of parameters. TODO: rename
    """
    if statistics.has_statistic(gap_name):
        if pre_computed_gap is None:
            with torch.no_grad():
                gap = sum([
                    torch.dist(a, b, p=2).item() for a, b in zip(
                        chain.from_iterable([[p for p in pg['params']]
                                             for pg in
                                             optimizer.param_groups]),
                        chain.from_iterable(real_theta))
                ])
        else:
            gap = pre_computed_gap

        statistics.update_on_batch(gap_name, gap, 1)