from torch.nn import Module
from torch.optim import Optimizer

from .interface import ScheduledOptimizationStepMultiPartitionTrainer
from pipe.pipeline.trainers.statistics import GlueStats, Stats


class GlueTrainer(ScheduledOptimizationStepMultiPartitionTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, model: Module,
                 optimizer: Optimizer,
                 scheduler,
                 statistics: Stats,
                 step_every=1):
        super().__init__(model,
                         optimizer,
                         scheduler,
                         statistics)
        # HACK: set by dataset.
        self.features = None
        self.num_labels = None
        self.loss_fn = None

        self.step_every = step_every

    def calc_test_stats(
            self,
            x,
            labels,
            # example_indices=None,  we know them, its sequential!
            batch_size=None):
        # NOTE: we include loss for dev, huggingface does not
        loss = self.loss_fn(x, labels)
        self.statistics: GlueStats
        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.predictions.append(x.detach())
        self.statistics.label_ids.append(labels.detach())

    def backprop_last_partition(self, x, labels, batch_size):
        loss = self.loss_fn(x, labels)  # FIXME...
        if self.step_every > 1:
            loss /= self.step_every
        loss.backward()
        return loss

    def last_partition_step_and_statistics(self,
                                           x,
                                           labels,
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
