import abc
import types
from functools import wraps


class GapAwareBase(abc.ABC):
    """
    GAP aware implementation for pipeline,
    based on
    https://arxiv.org/abs/1909.10802
    We can apply it if one of the following happends:
        1. we stash the parameters theta we did forwad on (so we could calculate the gap)
        2. the gap is easy (e.g the gradient)

    Notes:
        It adds memory footprint for SGD. (number of parameters in optimizer)

    Warning:
        Will not work with gradient accumulation as it changes grad !!!

        This implementation assumes staleness=1, so it should shut down for the first batch.


    Usage:
        
        call GapAwareBase.patch_scheduler(scheduler) to track max lr.

        After backward:

        update_running_stats()
        apply()  (if delay is > 0)

    Example:

        scheduler = ...
        optimizer = ...
        ga = ...

        ga.patch_scheduler(scheduler)

        loss = ...

        loss.backward()

        ga.update_running_stats()
        ga.apply()

        # Send gradients in pipeline

        optimizer.step()
        scheduler.step()

    TODO:
        Support working for all layers of pipeline
        Think about implementation with L2.
        Think about implementation with hooks. (can be tricky)
    """
    # 3 main changes form original implementation.
    # (1) better mem trick for WD.
    # (2) option to shut down changing WD. => but we came to conclusion its not recommended.
    #
    #
    # (this note is not correct...) Note: (2) will allow checking L2 regularization instead of WD.
    #    (as WD suffers from staleness)
    #
    # (this note is not correct: its accumulated to local parameters, we can't send it backwards.)
    # (3) For pipeline, we can send the "mitigated" gradients backwards, (before step),
    #       And thus avoid using GA on all layers.
    #       For WD mem trick, this requires applying a WD correction later.
    #       Better used with SGD+L2 than WD.

    MAX_LR_NAME = "max_lr"

    def __init__(self, optimizer):
        """ Apply Gap Aware on computed gradients """

        for pg in optimizer.param_groups:
            pg[GapAwareBase.MAX_LR_NAME] = pg['lr']

        self.optimizer = optimizer

        # FIXME can be of optimizer. e.g in adam its param_group['step']
        self.step_count = 0  # Need to be ahead of the optimizer by 1.

    def inc_step_count(self):
        self.step_count += 1

    def update_running_avg(self):
        """in case there is some running avg to update"""
        pass

    def update_running_stats(self):
        # TODO
        """ Basic method for updating running statistics """
        self.update_running_avg()
        self.inc_step_count()

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

    @staticmethod
    def patch_scheduler(scheduler):
        def step_decorator(func):
            @wraps(func)
            def inner(self, *args, **kwargs):
                func(self, *args, **kwargs)
                for pg in self.optimizer.param_groups:
                    # pg['max_lr'] = max(pg['max_lr'], pg['lr'])
                    pg[GapAwareBase.MAX_LR_NAME] = max(
                        pg[GapAwareBase.MAX_LR_NAME], pg['lr'])

            return types.MethodType(inner, scheduler)

        scheduler.step = step_decorator(scheduler.step.__func__)
        print(
            f"Scheduler.step() patched to also track max lr in pg[{GapAwareBase.MAX_LR_NAME}]"
        )
