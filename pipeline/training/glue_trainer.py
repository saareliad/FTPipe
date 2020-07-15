from .interface import BaseOutPutIsLossTrainer, BaseLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase

# HACK we layzyly use BaseOutPutIsLossTrainer
class GlueTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # NOTE: set by dataset
        self.features = None
        self.num_labels = None  # HACK: set by dataset.

        self.loss_fn = None

    def calc_test_stats(
            self,
            x,
            labels,
            # example_indices=None,  we know them, its sequential!
            batch_size=None):

        # NOTE: we include loss for dev, huggingface does not
        loss = self.loss_fn(x, labels)
        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.predictions.append(x.detach())
        self.statistics.label_ids.append(labels.detach())

    def backprop_last_partition(self, x, labels, batch_size):
        # logits = x[0]
        loss = self.loss_fn(x, labels)  # FIXME...
        # print(loss)
        return super().backprop_last_partition(loss)
        # if self.step_every > 1:
        #     loss /= self.step_every
        # loss.backward()
        # return loss

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


class GapAwareGlueTrainer(GlueTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        GlueTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)
