import torch
from itertools import chain
from .interface import GapAwareBase

# TODO: check that the step count is indeed OK (and ahead by 1)

class AdamGapAware(GapAwareBase):
    """ Gap aware for ADAM optimizer """

    def __init__(self, optimizer, big_gamma=0.999, epsilon=1e-8, from_grad=True):
        """ Apply Gap Aware on computed gradients """
        super().__init__(optimizer)



    @abc.abstractmethod
    def apply_from_grad(self):
        """ Calculate gap aware from gradient. Requires knowing the exact gap """
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_on_stashed(self, stashed_theta):
        """ True weights are loaded into the model, and given a stashed theta """
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_on_theta(self, real_theta):
        raise NotImplementedError()

    
    def update_running_avg(self):
        """in case there is some running avg to update"""
        pass




def get_adam_gap_aware_cls() -> GapAware:
    gap_aware_cls = AdamGapAware
    return gap_aware_cls
