import torch
from torch.nn import Module
from torch.optim import Optimizer

from .interface import ScheduledOptimizationStepMultiPartitionTrainer
# TODO: typehint for statistics. maybe it should actually sit under statistics
from pipe.pipeline.trainers.statistics import Stats


class CVTrainer(ScheduledOptimizationStepMultiPartitionTrainer):
    PER_STEP_SCHEDULER = False

    def __init__(self, model: Module,
                 optimizer: Optimizer,
                 scheduler,
                 statistics: Stats,
                 step_every=1):

        super(CVTrainer, self).__init__(model, optimizer, scheduler, statistics)
        self.step_every = step_every
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def backprop_last_partition(self, x, y, *args, **kw):
        loss = self.loss_fn(x, y)
        if self.step_every > 1:
            loss /= self.step_every
        loss.backward()
        return loss

    def calc_test_stats(self, x, y, batch_size):
        loss = self.loss_fn(x, y)
        assert (batch_size == len(y))
        y_pred = torch.argmax(x, 1)
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
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()

        if step:
            self.step_on_computed_grads(old_lrs)

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)


class CVTrainerPerStep(CVTrainer):
    PER_STEP_SCHEDULER = True

# TODO: it is also possible to do the entire thing on activation gradients,
#  avoiding the need to do it over all gradeints.