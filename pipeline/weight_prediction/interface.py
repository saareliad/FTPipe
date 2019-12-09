import abc
from typing import Callable


class WeightPredictor(abc.ABC):
    def __init__(self, optimizer,
                 fix_fn: Callable, scheduler=None):
        self.optimizer = optimizer
        # self.params = self.optimizer.param_groups[0]['params']
        self.fix_fn = fix_fn  # ()
        self.scheduler = scheduler

    def setup(self, n_steps):
        self.n_steps = n_steps

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def revert(self):
        raise NotImplementedError()


class FixFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p: WeightPredictor, pg):
        # WeightPredictor is used mainly to get sched from....
        raise NotImplementedError()
