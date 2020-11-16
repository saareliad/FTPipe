import torch



# TODO: should remove register from init after moving
from torch.nn import Module
from torch.optim import Optimizer

from pipe.pipeline.trainers.statistics import Stats
from pipe.pipeline.trainers.interface import ScheduledOptimizationStepMultiPartitionTrainer


class CEPTrainer(ScheduledOptimizationStepMultiPartitionTrainer):
    PER_STEP_SCHEDULER = False

    def __init__(self, model: Module,
                 optimizer: Optimizer,
                 scheduler,
                 statistics: Stats,
                 step_every=1):
        super().__init__(model, optimizer, scheduler, statistics)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.step_every = 1

    def calc_test_stats(self, x, y, batch_size):
        # print("Called calc_test_stats")
        loss = self.loss_fn(x, y)
        assert (batch_size == len(y))
        y_pred = torch.ge(x, 0.5)
        num_correct = torch.sum(y == y_pred).item()

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)

    def last_partition_step_and_statistics(self, x, y, batch_size, loss, step=True, old_lrs=None):
        """
        step
        stats

        step can be used later for grad accumulations
        """

        assert (batch_size == len(y))
        y_pred = torch.ge(x, 0.5)
        num_correct = torch.sum(y == y_pred).item()

        if step:
            self.step_on_computed_grads(old_lrs)

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)

    def backprop_last_partition(self, x, y, *args, **kw):
        loss = self.loss_fn(x, y)
        if self.step_every > 1:
            loss /= self.step_every
        loss.backward()
        return loss