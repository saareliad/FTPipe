import abc
from itertools import chain
import torch

# class LossTrainer(abc.ABC):
#     # TODO: for models which returns only loss...
#     pass


class AnyTrainer(abc.ABC):

    @abc.abstractmethod
    def backprop_last_partition(self, *args, **kw):
        pass

    @abc.abstractmethod
    def last_partition_step_and_statistics(self, *args, **kw):
        pass

    @abc.abstractmethod
    def step_on_computed_grads(self, **kw):
        pass

    @abc.abstractmethod
    def calc_test_stats(self, *args, **kw):
        pass

    def modify_gradients(self, *args, **kw):
        pass


class SupervisedTrainer(AnyTrainer):
    # added *args just to make it a true subtype.

    @abc.abstractmethod
    def backprop_last_partition(self, x, y, *args, **kw):
        pass

    @abc.abstractmethod
    def last_partition_step_and_statistics(self, x, y, *args, **kw):
        """
        Usually used for the last partiton (or any other partiton were x,y are needed)
        to calculate loss, gradients and do training steps

        We currently assume its the last partition for simplicity
        """
        pass


class PartitionedTrainer(AnyTrainer):
    @abc.abstractmethod
    def non_last_partition_step(self, *args, **kw):
        pass

    @staticmethod
    def calc_gap(i1, i2, p=2):
        # i1 = chain.from_iterable([[p for p in pg['params']] for pg in self.trainer.optimizer.param_groups])
        # i2 = chain.from_iterable(real_theta)

        with torch.no_grad():
            total_norm = sum([torch.dist(a, b, p=p).item()
                              for a, b in zip(i1, i2)])

            # total_norm = torch.stack([torch.dist(a, b, p=p) for a, b in zip(i1, i2)]).sum().item()

        return total_norm

    def try_record_real_gap_from_current(self, real_theta):
        if self.statistics.has_statistic("gap"):
            with torch.no_grad():
                gap = sum([torch.dist(a, b, p=2).item() for a, b in
                           zip(chain.from_iterable([[p for p in pg['params']]
                                                    for pg in self.optimizer.param_groups]),
                               chain.from_iterable(real_theta))])

            self.statistics.update_statistic_after_batch("gap", gap)


class PartitionedSupervisedTrainer(PartitionedTrainer, SupervisedTrainer):
    pass
