import abc
from typing import List

# class LossTrainer(abc.ABC):
#     # TODO: for models which returns only loss...
#     pass


class AnyTrainer(abc.ABC):
    @abc.abstractmethod
    def do_your_job(self, *args, **kw):
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
    def do_your_job(self, x, y, *args, **kw):
        """
        Usually used for the last partiton (or any other partiton were x,y are needed)
        to calculate loss, gradients and do training steps

        We currently assume its the last partition for simplicity
        """
        pass
