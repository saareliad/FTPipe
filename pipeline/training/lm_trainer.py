# import torch
from .interface import BaseOutPutIsLossTrainer
# TODO: typehint for statistics. maybe it should actually sit under statistics


class LMTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def calc_test_stats(self, loss, batch_size):

        loss = loss.item()
        self.statistics.update_on_batch("loss", loss, batch_size)
        self.statistics.update_on_batch("ppl", loss, batch_size)

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
        # Same as loos, we just get the statisitc differently...
        self.statistics.update_on_batch("ppl", loss, batch_size)