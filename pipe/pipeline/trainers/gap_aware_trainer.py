from typing import Type

from .interface import ScheduledOptimizationStepMultiPartitionTrainer
from ..gap_aware.interface import GapAwareBase

PipelineSupportedTrainerWithoutGapAware = ScheduledOptimizationStepMultiPartitionTrainer


class GapAwareTrainerMixin:
    HAS_GAP_AWARE = True

    def __init__(self, gap_aware: GapAwareBase, scheduler=None):
        self.gap_aware = gap_aware

        # Patch to update max_lr.
        if scheduler is not None:
            gap_aware.patch_scheduler(scheduler)

    def apply_gap_aware(self, real_theta=None, delay=None, stashed_theta=None):
        """ NOTE: we assume that if `real_theta` is given, a stashed weight is loaded into the model
        Otherwise, if stashed theta is given, we assume that the true weights are already loaded into the model,
        and we compute the gap from the stashed weights (used in "Gap aware just for loss" algorithm.
         """
        ga = self.gap_aware
        # NOTE: if they are not already record otherwise,
        # running statistics should record the step size per parameter and step count
        ga.update_running_stats()  # NOTE: SGD, like paper's implementation
        if delay:
            if real_theta:
                ga.apply_on_theta(real_theta)
            elif stashed_theta:
                ga.apply_on_stashed(stashed_theta)
            else:
                # TODO: note this should be called only before step, assuming delay of exactly 1.
                # FIXME: its very problematic if almost last partition calls this if step_every > 1.
                # This means: for the "gap_aware.json" configs !!!
                assert delay == 1
                ga.apply_from_grad()

def gap_aware_trainer_factory(trainer_cls: Type[PipelineSupportedTrainerWithoutGapAware]):
    class GapAwareCreatedTrainer(trainer_cls, GapAwareTrainerMixin):
        def __init__(self, gap_aware, scheduler=None, **kw):
            trainer_cls.__init__(self, scheduler=scheduler, **kw)
            GapAwareTrainerMixin.__init__(self, gap_aware, scheduler=scheduler)

    return GapAwareCreatedTrainer
