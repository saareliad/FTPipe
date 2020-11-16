from torch.nn import Module
from torch.optim import Optimizer

from .interface import ScheduledOptimizationStepMultiPartitionTrainer
from pipe.pipeline.trainers.statistics import Stats


class T5Trainer(ScheduledOptimizationStepMultiPartitionTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, model: Module,
                 optimizer: Optimizer,
                 scheduler,
                 statistics: Stats,
                 step_every=1,
                 loss_multiplier=1):
        super().__init__(model, optimizer, scheduler, statistics)
        self.step_every = step_every
        # when doing our type of packing vs T5 it loss_multiplier has effect (e.g batch 256 instead of 8)
        # set it to amount of packing we do.
        self.loss_multiplier = loss_multiplier

    def calc_test_stats(self, x, batch_size=None):
        # Eval in T5 happens offline,
        # loss is pretty much meaningless, calculated as (not a very good) sanity check.
        loss = x
        self.statistics.update_on_batch("loss", loss.item(), batch_size)

    def backprop_last_partition(self, x, batch_size):
        loss = x
        loss_multiplier = self.loss_multiplier
        if loss_multiplier != 1:
            loss *= loss_multiplier

        if self.step_every > 1:
            loss /= self.step_every
        loss.backward()
        return loss

    def last_partition_step_and_statistics(self,
                                           x,
                                           batch_size,
                                           loss,
                                           step=True,
                                           old_lrs=None):
        """
        x: is model output.
        
        step
        stats

        step can be used later for grad accumulations
        """
        if step:
            self.step_on_computed_grads(old_lrs)

        loss = loss.item()
        self.statistics.update_on_batch("loss", loss, batch_size)
