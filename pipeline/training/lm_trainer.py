import torch
from .interface import BaseLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase
# TODO: typehint for statistics. maybe it should actually sit under stats


class LMTrainer(BaseLossTrainer):
    def __init__(self, *args, **kw):
        super().__init__(*args, loss_fn=torch.nn.CrossEntropyLoss(), **kw)

    def calc_test_stats(self, x, y):
        # print("Called calc_test_stats")
        loss = self.loss_fn(x, y)
        batch_size = len(y)
        # acc = num_correct / batch_size
        self.statistics.on_batch_end(loss.item(), batch_size)

    def last_partition_step_and_statistics(self, x, y, loss, step=True):
        """
        step
        stats

        step can be used later for grad accumulations
        """

        batch_size = len(y)

        max_grad_norm = None
        if step:
            max_grad_norm = self.step_on_computed_grads()

        if max_grad_norm:  # Handles different classes of statistics. not so nice, should be fixed
            self.statistics.on_batch_end(loss.item(), batch_size,
                                         max_grad_norm)
        else:
            self.statistics.on_batch_end(loss.item(), batch_size)


class GapAwareLMTrainer(LMTrainer, GapAwareTrainerBase):
    # FIXME
    # HAS_GAP_AWARE = True
    def __init__(self, gap_aware, **kw):
        super(GapAwareLMTrainer, self).__init__(**kw)
        GapAwareTrainerBase.__init__(self, gap_aware)

        self.gap_aware = gap_aware

    def last_partition_step_and_statistics(self, x, y, loss, step=True):
        """
        step
        stats

        step can be used later for grad accumulations
        """
        # self.gap_aware.try_apply_wd_correction_before_step()
        super(LMTrainer, self).last_partition_step_and_statistics(x,
                                                                  y,
                                                                  loss,
                                                                  step=step)
        # TODO: self.ga.update_max_lr() add when we have per step scheduler
