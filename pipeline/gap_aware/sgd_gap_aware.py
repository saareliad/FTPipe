import torch
from itertools import chain
from .interface import GapAwareBase

class GapAware(GapAwareBase):
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

        This implementaion assumes staleness=1, so it should shut down for the first batch.


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
        Think about implmentation with L2.
        Think about implementation with hooks. (can be tricky)
    """
    # 3 main changes form original implementation.
    # (1) better mem trick for WD.
    # (2) option to shut down changing WD. => but we came to conclusion its not recomended.
    #
    #
    # (this note is not correct...) Note: (2) will allow checking L2 regularization instead of WD.
    #    (as WD suffers from staleness)
    #
    # (this note is not correct: its acumulated to local parameters, we can't send it backwards.)
    # (3) For pipeline, we can send the "mitigated" gradeints backwards, (before step),
    #       And thus avoid using GA on all layers.
    #       For WD mem trick, this requires applying a WD correction later.
    #       Better used with SGD+L2 than WD.

    def __init__(self, optimizer, big_gamma=0.999, epsilon=1e-8, from_grad=True):
        """ Apply Gap Aware on computed gradients """
        super().__init__(optimizer)
        if from_grad:
            assert type(optimizer) == torch.optim.SGD

        self.big_gamma = big_gamma  # FIXME can be of optimizer of given. e.g adam

        # FIXME can be of optimizer of given. e.g adam
        # Iter over optimizer parameters:
        opt_params_iter = chain(*[pg['params']
                                  for pg in optimizer.param_groups])
        self.running_avg_step = {id(p): torch.zeros_like(p)
                                 for p in opt_params_iter}

        # FIXME can be of optimizer. e.g in adam its param_group['step']
        # self.step_count = 0
        self.epsilon = epsilon  # FIXME can be of optimizer.

        # Ugly hack, init momentum buffer to zeros before we start
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                if 'momentum_buffer' not in self.optimizer.state[p]:
                    self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

    def update_running_avg(self):
        """
        Update the exponential step running average
        Requires: that we got some grad.
        """
        # For SGD...
        # Note: its pow 2 because we later do pow 0.5
        opt_s = self.optimizer.state
        ra = self.running_avg_step
        bg = self.big_gamma
        with torch.no_grad():
            for pg in self.optimizer.param_groups:
                if pg['momentum'] != 0:
                    for p in pg['params']:
                        ra[id(p)].data = bg * ra[id(p)].data + \
                            (1 - bg) * \
                            (opt_s[p]["momentum_buffer"].data ** 2)
                else:
                    for p in pg['params']:
                        ra[id(p)].data = bg * ra[id(p)].data + \
                            (1 - bg) * ((p.grad.data) ** 2)

    def apply_from_grad(self):
        """ Calculate gap aware from gradient. Requires knowing the exact gap """
        with torch.no_grad():
            ra = self.running_avg_step
            bias_correction = 1 - (self.big_gamma ** self.step_count)
            eps = self.epsilon
            # Calculate gap from grad
            for pg in self.optimizer.param_groups:
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                for p in pg['params']:
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((ra[id(p)].data / bias_correction) ** 0.5) + eps)

                    # calculate the gap per-element
                    penalty = 1 + (pg['lr'] * p.grad.abs() / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # HACK: we know that sgd does
                    #   d_p += p*wd
                    # and we want:
                    #   d_p += p*wd/penalty
                    # so we solve:
                    # x + z + p*wd = x + (p*wd / penalty)
                    # giving:
                    # z = p*wd ((1/penalty) - 1) = ((1 - penalty) / penalty)
                    # so we do
                    #   d_p += z
                    # z =  p.data * weight_decay * ((1 - penalty) / penalty)
                    p.grad.data += p.data.mul(weight_decay *
                                              ((1 - penalty) / penalty))

    def apply_on_theta(self, real_theta):
        with torch.no_grad():
            ra = self.running_avg_step
            bias_correction = 1 - (self.big_gamma ** self.step_count)
            eps = self.epsilon
            # Calculate gap from grad
            for pg, rpg in zip(self.optimizer.param_groups, real_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                for p, rp in zip(pg['params'], rpg):
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((ra[id(p)].data / bias_correction) ** 0.5) + eps)

                    gap = (p - rp).abs()
                    # pg['lr'] * p.grad.abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # HACK: we know that sgd does
                    #   d_p += p*wd
                    # and we want:
                    #   d_p += p*wd/penalty
                    # so we solve:
                    # x + z + p*wd = x + (p*wd / penalty)
                    # giving:
                    # z = p*wd ((1/penalty) - 1) = ((1 - penalty) / penalty)
                    # so we do
                    #   d_p += z
                    # z =  p.data * weight_decay * ((1 - penalty) / penalty)

                    # NOTE: we apply the weight decay on the real parameter weight, rp.
                    p.grad.data += rp.data.mul(weight_decay *
                                               ((1 - penalty) / penalty))

    def apply_on_stashed(self, stashed_theta):
        """ True weights are loaded into the model, and given a stashed theta """
        with torch.no_grad():
            ra = self.running_avg_step
            bias_correction = 1 - (self.big_gamma ** self.step_count)
            eps = self.epsilon
            # Calculate gap from grad
            for pg, spg in zip(self.optimizer.param_groups, stashed_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                for p, sp in zip(pg['params'], spg):
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((ra[id(p)].data / bias_correction) ** 0.5) + eps)

                    gap = (p - sp).abs()
                    # pg['lr'] * p.grad.abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # HACK: we know that sgd does
                    #   d_p += p*wd
                    # and we want:
                    #   d_p += p*wd/penalty
                    # so we solve:
                    # x + z + p*wd = x + (p*wd / penalty)
                    # giving:
                    # z = p*wd ((1/penalty) - 1) = ((1 - penalty) / penalty)
                    # so we do
                    #   d_p += z
                    # z =  p.data * weight_decay * ((1 - penalty) / penalty)

                    # NOTE: we apply the weight decay on the real parameter weight, rp.
                    p.grad.data += p.data.mul(weight_decay *
                                              ((1 - penalty) / penalty))


# FIXME: keys are hardcoded from optimizers...
SGD_TYPE_TO_GAP_AWARE_CLASS = {
    'sgd1': GapAware,  # Pytorch
    # 'sgd2':   # TF # TODO
}


def get_sgd_gap_aware_cls(sgd_type: str) -> GapAware:
    gap_aware_cls = SGD_TYPE_TO_GAP_AWARE_CLASS.get(sgd_type, None)
    return gap_aware_cls
    # return gap_aware_cls(*args, **kw)


# class GAStats:
