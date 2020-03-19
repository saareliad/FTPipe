# import torch
from .interface import BaseOutPutIsLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase
import math
# TODO: typehint for statistics. maybe it should actually sit under stats


class LMTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def calc_test_stats(self, loss, batch_size):

        loss = loss.item()
        self.statistics.update_on_batch("loss", loss, batch_size)
        self.statistics.update_on_batch("ppl", math.exp(loss), batch_size)

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
        if step:
            self.step_on_computed_grads()

        loss = loss.item()
        self.statistics.update_on_batch("loss", loss, batch_size)
        self.statistics.update_on_batch("ppl", math.exp(loss), batch_size)


class GapAwareLMTrainer(LMTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        # super(GapAwareLMTrainer, self).__init__(**kw)
        LMTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)