from .interface import PartitionedTrainer


class GapAwareTrainerBase(PartitionedTrainer):
    HAS_GAP_AWARE = True

    def __init__(self, gap_aware, scheduler=None):
        super().__init__()
        self.gap_aware = gap_aware

        # Patch to update max_lr.
        if scheduler is not None:
            gap_aware.patch_scheduler(scheduler)

    def modify_gradients(self, real_theta=None, delay=None,
                         stashed_theta=None):
        """ NOTE: we assume that if `real_theta` is given, a stashed weight is loaded into the model
        Otherwise, if stashed theta is given, we assume that the true weights are already loaded into the model,
        and we compute the gap from the stashed weights (used in "Gap aware just for loss" algorithm.
         """
        # TODO: we may want to save some statistics before we modify grad.
        ga = self.gap_aware
        ga.update_running_stats()
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
