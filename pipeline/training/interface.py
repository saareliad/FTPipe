import abc
from itertools import chain
import torch
from torch.nn.utils import clip_grad_norm_
from .utils import calc_norm
from ..statistics.interface import Stats
from torch.optim.optimizer import Optimizer
from torch.nn import Module


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


class SupervisedLossIncludedTrainer(AnyTrainer):
    def backprop_last_partition(self, loss, *args, **kw):
        if self.step_every > 1:
            loss /= self.step_every
        loss.backward()
        return loss


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
    # def __init__(self, optimizer, statistics):
    #     self.optimizer = optimizer
    #     self.statistics = statistics

    @abc.abstractmethod
    def non_last_partition_step(self, *args, **kw):
        pass

    def try_record_real_gap_from_current(self,
                                         real_theta,
                                         pre_computed_gap=None,
                                         gap_name="gap"):
        """ calculates gap between model parameters and a given set of parameters, real_theta
            real_theta: Given set of parameters. TODO: rename
        """
        # TODO: this is very weird from here. this is certainly not the place.
        # TODO: should pass a function to calculate this and do it in the right place.
        if self.statistics.has_statistic(gap_name):
            if pre_computed_gap is None:
                with torch.no_grad():
                    gap = sum([
                        torch.dist(a, b, p=2).item() for a, b in zip(
                            chain.from_iterable([[p for p in pg['params']]
                                                 for pg in
                                                 self.optimizer.param_groups]),
                            chain.from_iterable(real_theta))
                    ])
            else:
                gap = pre_computed_gap

            # FIXME:
            self.statistics.update_on_batch(gap_name, gap, 1)


class PartitionedSupervisedTrainer(PartitionedTrainer, SupervisedTrainer):
    pass


class PartitionedSupervisedLossIncludedTrainer(PartitionedTrainer,
                                               SupervisedLossIncludedTrainer):
    pass


class GradNormStepper:
    PER_STEP_SCHEDULER = False

    # def __init__(
    #         self,
    #         model,
    #         optimizer,
    #         scheduler,
    #         statistics,
    #         max_grad_norm=None,
    #         always_calc_grad_norm=False,
    # ):
    #     self.optimizer = optimizer
    #     self.scheduler = scheduler
    #     self.model = model
    #     self.max_grad_norm = max_grad_norm
    #     self.always_calc_grad_norm = always_calc_grad_norm
    #     self.statistics = statistics
    def non_last_partition_step(self, old_lrs=None):
        self.step_on_computed_grads(old_lrs=old_lrs)

    def step_on_computed_grads(self, old_lrs=None):
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Restore old LRs, to avoid messing up scheduler.
        if old_lrs:
            pgs = self.optimizer.param_groups
            for g, old_lr in zip(pgs, old_lrs):
                g['lr'] = old_lr

        if self.PER_STEP_SCHEDULER:
            self.scheduler.step()

    def grad_norm(self):
        # Grad norm
        max_grad_norm = None
        if self.max_grad_norm:
            with torch.no_grad():
                max_grad_norm = clip_grad_norm_(self.model.parameters(),
                                                self.max_grad_norm,
                                                norm_type=2)
        elif self.always_calc_grad_norm:
            with torch.no_grad():
                max_grad_norm = calc_norm(self.model.parameters(), norm_type=2)

        if max_grad_norm:
            self.statistics.update_on_batch("grad_norm", max_grad_norm, 1)


class BaseLossTrainer(GradNormStepper, PartitionedSupervisedTrainer):
    """Trainer assuming loss is calculated *outside* the model """
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 scheduler,
                 statistics: Stats,
                 max_grad_norm=None,
                 always_calc_grad_norm=False,
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 cuda=True,
                 step_every=1):

        self.loss_fn = loss_fn
        if cuda:
            self.loss_fn = self.loss_fn.cuda()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.always_calc_grad_norm = always_calc_grad_norm
        self.step_every = step_every

        # Stats
        self.statistics = statistics

    def backprop_last_partition(self, x, y):
        loss = self.loss_fn(x, y)
        if self.step_every > 1:
            loss /= self.step_every
        loss.backward()  # this does backward() only for the last partition
        return loss


class BaseOutPutIsLossTrainer(GradNormStepper,
                              PartitionedSupervisedLossIncludedTrainer):
    """Trainer assuming loss is calculated *inside* the model """
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 scheduler,
                 statistics: Stats,
                 max_grad_norm=None,
                 always_calc_grad_norm=False,
                 step_every=1):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.always_calc_grad_norm = always_calc_grad_norm
        self.step_every = step_every

        # Stats
        self.statistics = statistics
