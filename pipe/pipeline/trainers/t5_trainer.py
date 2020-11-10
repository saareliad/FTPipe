from .interface import LossIncludedInModelMultiPartitionTrainer


class T5Trainer(LossIncludedInModelMultiPartitionTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, loss_multiplier=1, **kw):
        super().__init__(*args, **kw)
        # when doing our type of packing vs T5 it loss_multiplier has effect (e.g batch 256 instead of 8)
        # set it to amount of packing we do.
        self.loss_multiplier = loss_multiplier
        print(f"-I- trainer: got loss_multiplier={self.loss_multiplier}")

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
        loss = x
        loss_multiplier = self.loss_multiplier
        if loss_multiplier != 1:
            loss *= loss_multiplier
        return super().backprop_last_partition(loss)

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
