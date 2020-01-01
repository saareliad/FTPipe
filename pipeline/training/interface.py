import abc

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


class PartitionedSupervisedTrainer(PartitionedTrainer, SupervisedTrainer):
    pass
