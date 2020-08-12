from .interface import BaseOutPutIsLossTrainer, BaseLossTrainer


class SquadTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # NOTE: set by dataset
        self.features = None

    def advanced_test_stats(self, x, example_indices):
        raise NotImplementedError()

    def calc_test_stats(self, x, batch_size=None):

        # NOTE: we include loss for dev, huggingface does not
        loss = x
        self.statistics.update_on_batch("loss", loss.item(), batch_size)

        # TODO: this happens in eval only.
        # if example_indices is not None:
        #    self.advanced_test_stats(x, example_indices)
    
    def backprop_last_partition(self, x, batch_size):
        # logits = x[0]
        loss = x
        return super().backprop_last_partition(loss)
        # if self.step_every > 1:
        #     loss /= self.step_every
        # loss.backward()
        # return loss

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