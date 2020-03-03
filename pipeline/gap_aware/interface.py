import abc

class GapAwareBase(abc.ABC):
    """
    GAP aware implementation for pipeline,
    based on
    https://arxiv.org/abs/1909.10802
    We can apply it if one of the following happends:
        1. we stash the parameters theta we did forwad on (so we could calculate the gap)
        2. the gap is easy (e.g the gradient)

    Notes:
        It adds memory footprint. (number of parameters in optimizer)

    Warning:
        Will not work with gradient accumulation as it changes grad !!!

        This implementation assumes staleness=1, so it should shut down for the first batch.


    Usage:

        After backward:
            update_running_avg()

        Before apply:
            inc_step_count()

        # FIXME: deprecated docstring
        # # Apply on gradients:
        #     apply_grad_only()

        #     WARNINING: MUST HAVE A CORRESPONDING CALL TO try_apply_wd_correction_before_step()

        # Before optimizer.step():
        #     try_apply_wd_correction_before_step()

        apply()

        After each scheduler.step():
            update_max_lr()

    Note:
        For non-pipline settings, just call apply() instad of two sperate calles.

    Example:

        scheduler = ...
        optimizer = ...
        ga = ...
        loss = ...

        loss.backward()

        ga.update_running_avg()
        ga.inc_step_count()
        ga.apply()

        # Send gradients in pipeline

        optimizer.step()
        scheduler.step()

        ga.update_max_lr()

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

        self.optimizer = optimizer

        # FIXME can be of optimizer. e.g in adam its param_group['step']
        self.step_count = 0   # Need to be ahead of the optimizer on 1.

    def update_max_lr(self):
        """ should be called after scheduler step. """
        for pg in self.optimizer.param_groups:
            pg[GapAwareBase.MAX_LR_NAME] = max(pg[GapAwareBase.MAX_LR_NAME], pg['lr'])

    def inc_step_count(self):
        self.step_count += 1


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