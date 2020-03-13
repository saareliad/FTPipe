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
        # acc = num_correct / batch_size
        self.statistics.last_partition_on_batch_end(loss.item(), num_correct, batch_size)

    def last_partition_step_and_statistics(self, x, y, loss, step=True):
        """
        step
        stats

        step can be used later for grad accumulations
        """

        batch_size = len(y)
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()

        max_grad_norm = None
        if step:
            max_grad_norm = self.step_on_computed_grads()

        if max_grad_norm:  # Handles different classes of statistics. not so nice, should be fixed
            self.statistics.last_partition_on_batch_end(loss.item(), num_correct, batch_size,
                                         max_grad_norm)
        else:
            self.statistics.last_partition_on_batch_end(loss.item(), num_correct, batch_size)


# TODO: it is also possible to do the entire thing on activation gradients,
#  avoiding the need to do it over all gradeints.


class GapAwareCVTrainer(CVTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        # super(GapAwareCVTrainer, self).__init__(**kw)
        CVTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)