import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import itertools


def dummy_optimizer(lr, n_param_groups=1):
    """ Dummy optimizer with dummy model """
    model = nn.Linear(1, 1, bias=False)
    optimizer = optim.SGD(model.parameters(), lr)

    for i in range(1, n_param_groups):
        model = nn.Linear(1, 1, bias=False)
        optimizer.add_param_group({'params': model.parameters()})
    return optimizer


class SchedulerPredictor:
    def __init__(self, lr, sched_creator_cls, *args, n_param_groups=0, **kw):
        optimizer = dummy_optimizer(lr=lr, n_param_groups=n_param_groups)
        scheduler = sched_creator_cls(optimizer, *args, **kw)
        optimizer.step()  # Dummy step to supress annoying warnings...
        self.scheduler = scheduler

        self.q = deque()
        self.q.append(self.scheduler.get_last_lr())

    def get_next(self, n_next, n_pop=1):

        while len(self.q) < n_next:
            self.scheduler.step()
            self.q.append(self.scheduler.get_last_lr())

        res = list(itertools.islice(self.q, 0, n_next))

        for _ in range(n_pop):
            self.q.popleft()

        return res


if __name__ == "__main__":

    from transformers import (
        # get_constant_schedule_with_warmup,
        # get_constant_schedule,
        get_linear_schedule_with_warmup,
        # get_cosine_schedule_with_warmup,
        # get_cosine_with_hard_restarts_schedule_with_warmup
    )

    d = {
        "lr": 0.1,
        "sched_creator_cls": get_linear_schedule_with_warmup,
        "n_param_groups": 1,
        "num_warmup_steps": 5,
        "num_training_steps": 10,
        "last_epoch": -1,
    }
    sp = SchedulerPredictor(**d)

    # This simulates running on steady state with staleness of 3
    # (e.g, partition with depth 3)
    print(sp.get_next(3))
    print(sp.get_next(3))
    print(sp.get_next(3))
    print(sp.get_next(3))
    print(sp.get_next(3))
