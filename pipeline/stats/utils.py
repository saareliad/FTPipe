from typing import Dict, NamedTuple
from types import SimpleNamespace


def fit_res_to_dict(fit_res) -> Dict:
    if isinstance(fit_res, NamedTuple):
        fit_res = fit_res._asdict()
    elif isinstance(fit_res, SimpleNamespace):
        fit_res = fit_res.__dict__
    # elif isinstance(fit_res, dict)
    #     pass

    return fit_res


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
    