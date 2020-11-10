import torch

from .interface import DataAndLabelsMultiPartitionTrainer


# TODO: should remove register from init after moving

class CEPTrainer(DataAndLabelsMultiPartitionTrainer):
    PER_STEP_SCHEDULER = False

    def __init__(self, *args, **kw):
        super().__init__(*args, loss_fn=torch.nn.BCEWithLogitsLoss(), **kw)

    def calc_test_stats(self, x, y, batch_size):
        # print("Called calc_test_stats")
        loss = self.loss_fn(x, y)
        assert (batch_size == len(y))
        y_pred = torch.ge(x, 0.5)
        num_correct = torch.sum(y == y_pred).item()

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)

    def last_partition_step_and_statistics(self, x, y, batch_size, loss, step=True, old_lrs=None):
        """
        step
        stats

        step can be used later for grad accumulations
        """

        assert (batch_size == len(y))
        y_pred = torch.ge(x, 0.5)
        num_correct = torch.sum(y == y_pred).item()

        if step:
            self.step_on_computed_grads(old_lrs)

        self.statistics.update_on_batch("loss", loss.item(), batch_size)
        self.statistics.update_on_batch("acc", num_correct, batch_size)
