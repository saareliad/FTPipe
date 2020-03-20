import torch
from .interface import BaseLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase
# TODO: typehint for statistics. maybe it should actually sit under stats


class CVTrainer(BaseLossTrainer):
    PER_STEP_SCHEDULER = False

    def __init__(self, *args, **kw):
        super().__init__(*args, loss_fn=torch.nn.CrossEntropyLoss(), **kw)

    def calc_test_stats(self, x, y):
        # print("Called calc_test_stats")
        loss = self.loss_fn(x, y)
        batch_size = len(y)
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)

    def last_partition_step_and_statistics(self, x, y, loss, step=True):
        """
        step
        stats

        step can be used later for grad accumulations
        """

        batch_size = len(y)
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()

        if step:
            self.step_on_computed_grads()

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)

# TODO: it is also possible to do the entire thing on activation gradients,
#  avoiding the need to do it over all gradeints.


class GapAwareCVTrainer(CVTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        # super(GapAwareCVTrainer, self).__init__(**kw)
        CVTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)