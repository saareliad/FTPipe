# import torch
from .interface import BaseOutPutIsLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase
# TODO: typehint for statistics. maybe it should actually sit under stats


class LMTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def calc_test_stats(self, loss, batch_size):
        # print("Called calc_test_stats")
        # loss = self.loss_fn(x, y)
        # batch_size = len(y)
        # acc = num_correct / batch_size
        self.statistics.last_partition_on_batch_end(loss.item(), batch_size)

    def last_partition_step_and_statistics(self,
                                           x,
                                           batch_size,
                                           loss,
                                           step=True):
        """
        x: is model output.
        
        step
        stats

        step can be used later for grad accumulations
        """

        max_grad_norm = None
        if step:
            max_grad_norm = self.step_on_computed_grads()

        if max_grad_norm:  # Handles different classes of statistics. not so nice, should be fixed
            self.statistics.last_partition_on_batch_end(loss.item(), batch_size,
                                         max_grad_norm)
        else:
            self.statistics.last_partition_on_batch_end(loss.item(), batch_size)


class GapAwareLMTrainer(LMTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        # super(GapAwareLMTrainer, self).__init__(**kw)
        LMTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)