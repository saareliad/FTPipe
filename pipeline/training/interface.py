import abc
from itertools import chain
import torch
from torch.nn.utils import clip_grad_norm_
from .utils import calc_norm

# class LossTrainer(abc.ABC):
#     # TODO: for models which returns only loss...
#     pass


class AnyTrainer(abc.ABC):
    @abc.abstractmethod
    def backprop_last_partition(self, *args, **kw):
        pass

    @abc.abstractmethod
    def last_partition_step_and_statistics(self, *args, **kw):
        pass

    @abc.abstractmethod
    def step_on_computed_grads(self, **kw):
        pass

    @abc.abstractmethod
    def calc_test_stats(self, *args, **kw):
        pass

    def modify_gradients(self, *args, **kw):
        pass


class SupervisedTrainer(AnyTrainer):
    # added *args just to make it a true subtype.

    @abc.abstractmethod
    def backprop_last_partition(self, x, y, *args, **kw):
        pass

    @abc.abstractmethod
    def last_partition_step_and_statistics(self, x, y, *args, **kw):
        """
        Usually used for the last partiton (or any other partiton were x,y are needed)
        to calculate loss, gradients and do training steps

        We currently assume its the last partition for simplicity
        """
        pass


class PartitionedTrainer(AnyTrainer):
    @abc.abstractmethod
    def non_last_partition_step(self, *args, **kw):
        pass

    def try_record_real_gap_from_current(self, real_theta):
        if self.statistics.has_statistic("gap"):
            with torch.no_grad():
                gap = sum([
                    torch.dist(a, b, p=2).item() for a, b in zip(
                        chain.from_iterable([[p for p in pg['params']] for pg
                                             in self.optimizer.param_groups]),
                        chain.from_iterable(real_theta))
                ])

            self.statistics.update_statistic_after_batch("gap", gap)


class PartitionedSupervisedTrainer(PartitionedTrainer, SupervisedTrainer):
    pass


class BaseLossTrainer(PartitionedSupervisedTrainer):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 statistics,
                 max_grad_norm=None,
                 always_calc_grad_norm=False,
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 cuda=True):

        self.loss_fn = loss_fn
        if cuda:
            self.loss_fn = self.loss_fn.cuda()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.ALWAYS_CALC_NORM = always_calc_grad_norm

        # Stats
        self.statistics = statistics

    def backprop_last_partition(self, x, y):
        loss = self.loss_fn(x, y)
        loss.backward()  # this does backward() only for the last partition
        return loss

    def non_last_partition_step(self):
        max_grad_norm = self.step_on_computed_grads()
        # Handles different classes of statistics. not so nice, should be fixed
        if not (max_grad_norm is None):
            self.statistics.non_last_partition_on_batch_end(max_grad_norm)

    def step_on_computed_grads(self):
        # TODO: implement gradient statistics later
        max_grad_norm = None
        if self.max_grad_norm:
            with torch.no_grad():
                max_grad_norm = clip_grad_norm_(self.model.parameters(),
                                                self.max_grad_norm,
                                                norm_type=2)
        elif self.ALWAYS_CALC_NORM:
            with torch.no_grad():
                max_grad_norm = calc_norm(self.model.parameters(), norm_type=2)

        self.optimizer.step()
        self.optimizer.zero_grad()
        # TODO: per step scheduler
        # self.scheduler.step()

        return max_grad_norm