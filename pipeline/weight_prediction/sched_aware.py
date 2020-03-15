import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import itertools

__all__ = ["get_sched_predictor", "SchedulerPredictor"]

def get_sched_predictor(optimizer, sched_creator_cls, **kw):
    """ Get scher predictor from optimizer and scheduler class and kwargs """
    n_param_groups = len(optimizer.param_groups)
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    d = {
        "lrs": lrs,
        "sched_creator_cls": sched_creator_cls,
        "n_param_groups": n_param_groups,
    }
    d = {**d, **kw}
    return SchedulerPredictor(**d)


def dummy_optimizer(lrs, n_param_groups=1):
    """ Dummy optimizer with dummy model """
    assert len(lrs) == n_param_groups
    model = nn.Linear(1, 1, bias=False)
    optimizer = optim.SGD(model.parameters(), lrs[0])

    for i in range(1, n_param_groups):
        model = nn.Linear(1, 1, bias=False)
        optimizer.add_param_group({'params': model.parameters(), 'lr': lrs[i]})
    return optimizer


class SchedulerPredictor:
    def __init__(self, lrs, sched_creator_cls, *args, n_param_groups=0, **kw):
        optimizer = dummy_optimizer(lrs=lrs, n_param_groups=n_param_groups)
        scheduler = sched_creator_cls(optimizer, *args, **kw)
        optimizer.step()  # Dummy step to supress annoying warnings...
        self.scheduler = scheduler

        self.q = deque()
        self.q.append(self.scheduler.get_last_lr())

    def get_next(self, n_next):

        while len(self.q) < n_next:
            self.scheduler.step()
            self.q.append(self.scheduler.get_last_lr())

        res = list(itertools.islice(self.q, 0, n_next))
        return res

    # def update_on_step(self):
    #     if len(self.q) > 0:
    #         self.q.popleft()

    def patch_scheduler(self, scheduler):
        q = self.q
        dummy_sched = self.scheduler
        def step_decorator(func):
            @wraps(func)
            def inner(self, *args, **kwargs):
                func(self, *args, **kwargs)

                q.append(dummy_sched.get_last_lr())
                q.popleft()

            return types.MethodType(inner, scheduler)

        scheduler.step = step_decorator(scheduler.step.__func__)
        print(
            f"-I- patched scheduler to update sched-aware predictor on step()"
        )



if __name__ == "__main__":

    from transformers import (
        # get_constant_schedule_with_warmup,
        # get_constant_schedule,
        get_linear_schedule_with_warmup,
        # get_cosine_schedule_with_warmup,
        # get_cosine_with_hard_restarts_schedule_with_warmup
    )

    d = {
        "lrs": [0.1],
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
