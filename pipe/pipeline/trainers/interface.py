import abc

from torch.optim.optimizer import Optimizer

from pipe.pipeline.trainers.statistics.interface import Stats


class LastPartitionTrainer(abc.ABC):
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



class DataAndLabelsLastPartitionTrainer(LastPartitionTrainer):
    """Adding x,y to represents (data,labels)."""

    @abc.abstractmethod
    def backprop_last_partition(self, x, y, *args, **kw):
        pass

    @abc.abstractmethod
    def last_partition_step_and_statistics(self, x, y, *args, **kw):
        """
        Usually used for the last partition (or any other partition were x,y are needed)
        to calculate loss, gradients and do training steps

        We currently assume its the last partition for simplicity
        """
        pass


class MultiPartitionTrainer(LastPartitionTrainer):
    def __init__(self, optimizer: Optimizer, statistics: Stats):
        self.optimizer = optimizer
        self.statistics = statistics

    @abc.abstractmethod
    def non_last_partition_step(self, *args, **kw):
        pass


class ScheduledOptimizationStepMultiPartitionTrainer(MultiPartitionTrainer):
    PER_STEP_SCHEDULER = False

    def __init__(self, model, optimizer, scheduler, statistics: Stats):
        super().__init__(optimizer, statistics)
        self.model = model
        self.scheduler = scheduler

    def non_last_partition_step(self, old_lrs=None):
        self.step_on_computed_grads(old_lrs=old_lrs)

    def step_on_computed_grads(self, old_lrs=None):
        self.optimizer.step()

        # self.optimizer.zero_grad()
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                p.grad = None

        # Restore old LRs, to avoid messing up scheduler.
        if old_lrs:
            pgs = self.optimizer.param_groups
            for g, old_lr in zip(pgs, old_lrs):
                g['lr'] = old_lr

        if self.PER_STEP_SCHEDULER:
            self.scheduler.step()
