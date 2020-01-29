import torch
from itertools import chain


class GapAware:
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

        This implementaion assumes staleness=1, so it should shut down for the first batch, with
            skip_one_apply()


    Usage:

        After backward:
            update_running_avg()

        Before apply:
            inc_step_count()

        # Apply on gradients:
            apply_grad_only()

            WARNINING: MUST HAVE A CORRESPONDING CALL TO try_apply_wd_correction_before_step()

        Before optimizer.step():
            try_apply_wd_correction_before_step()

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

    MAX_LR_NAME = "max_lr"

    def __init__(self, optimizer, big_gamma=0.999, epsilon=1e-8, penatly_for_weight_decay=True):
        """ set penatly_for_weight_decay=False when using without MSNAG """
        assert type(optimizer) == torch.optim.SGD

        # self.max_lr = max_lr
        self.optimizer = optimizer

        for pg in optimizer.param_groups:
            pg[self.MAX_LR_NAME] = pg['lr']

        self.big_gamma = big_gamma  # FIXME can be of optimizer of given. e.g adam

        # FIXME can be of optimizer of given. e.g adam
        # Iter over optimizer parameters:
        opt_params_iter = chain(*[pg['params']
                                  for pg in optimizer.param_groups])
        self.running_avg_step = {id(p): torch.zeros_like(p)
                                 for p in opt_params_iter}

        # FIXME can be of optimizer. e.g in adam its param_group['step']
        self.step_count = 0   # Need to be ahead of the optimizer on 1.
        self.epsilon = epsilon  # FIXME can be of optimizer.

        self.penatly_for_weight_decay = penatly_for_weight_decay
        self.skip_next_apply = True

        # Ugly hack, init momentum buffer to zeros before we start
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                if 'momentum_buffer' not in self.optimizer.state[p]:
                    self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(
                        p)

    def update_max_lr(self):
        """ should be called after scheduler step. """
        for pg in self.optimizer.param_groups:
            pg[self.MAX_LR_NAME] = max(pg[self.MAX_LR_NAME], pg['lr'])

    def skip_one_apply(self):
        self.skip_next_apply = True

    def inc_step_count(self):
        self.step_count += 1

    def update_running_avg(self):
        """
        Update the exponential step running average
        Requires: that we got some grad.
        """
        # For SGD...
        # Note: its pow 2 because we later do pow 0.5
        for pg in self.optimizer.param_groups:
            if pg['momentum'] != 0:
                for p in pg['params']:
                    self.running_avg_step[id(p)].data = self.big_gamma * self.running_avg_step[id(p)].data + \
                        (1 - self.big_gamma) * \
                        (self.optimizer.state[p]["momentum_buffer"].data ** 2)
            else:
                for p in pg['params']:
                    self.running_avg_step[id(p)].data = self.big_gamma * self.running_avg_step[id(p)].data + \
                        (1 - self.big_gamma) * ((p.grad.data) ** 2)

    # Note: I wanted to decorate the function, but seems like there is a bug
    # https://discuss.pytorch.org/t/combining-no-grad-decorator-and-with-torch-no-grad-operator-causes-gradients-to-be-enabled/39203
    # and i'm afraid of it...
    def apply(self, from_grad=True, on_grad=True, try_on_wd=True, ignore_skip_apply=False):
        assert (on_grad or try_on_wd)

        if self.skip_next_apply and not ignore_skip_apply:
            # Flip
            self.skip_next_apply = False

        if self.skip_next_apply:
            # Skip (not, we sometime only Skip, but don't Flip)
            return

        if (not on_grad) and (try_on_wd and (not self.penatly_for_weight_decay)):
            # nothing to do.
            return

        if not from_grad:
            raise NotImplementedError(
                "Gap claculation is supported only from grad")

        with torch.no_grad():
            bias_correction = 1 - (self.big_gamma ** self.step_count)
            # Calculate gap from grad
            for pg in self.optimizer.param_groups:
                if pg[self.MAX_LR_NAME] <= 0:
                    continue
                weight_decay = pg['weight_decay']
                for p in pg['params']:
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = pg[self.MAX_LR_NAME] * \
                        (((self.running_avg_step[id(
                            p)].data / bias_correction) ** 0.5) + self.epsilon)

                    # calculate the gap per-element
                    penalty = 1 + (pg['lr'] * p.grad.abs() / avg_steps_needed)

                    # Apply penalty to gradient
                    if on_grad:
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
                    if try_on_wd:
                        if self.penatly_for_weight_decay:
                            p.grad.data += p.data.mul(weight_decay *
                                                      ((1 - penalty) / penalty))

    def apply_on_theta(self, real_theta, delay, from_grad=False, on_grad=True, try_on_wd=True, ignore_skip_apply=False):
        assert (on_grad or try_on_wd)

        if self.skip_next_apply and not ignore_skip_apply:
            # Flip
            self.skip_next_apply = False

        if self.skip_next_apply:
            # Skip (not, we sometime only Skip, but don't Flip)
            return

        if (not on_grad) and (try_on_wd and (not self.penatly_for_weight_decay)):
            # nothing to do.
            return

        if from_grad:
            raise NotImplementedError(
                "Use the other function")

        with torch.no_grad():
            bias_correction = 1 - (self.big_gamma ** self.step_count)
            # Calculate gap from grad
            for pg, rpg in zip(self.optimizer.param_groups, real_theta):
                if pg[self.MAX_LR_NAME] <= 0:
                    continue
                weight_decay = pg['weight_decay']
                for p, rp in zip(pg['params'], rpg):
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = pg[self.MAX_LR_NAME] * \
                        (((self.running_avg_step[id(
                            p)].data / bias_correction) ** 0.5) + self.epsilon)

                    avg_steps_needed *= delay
                    gap = (p - rp).abs()
                    # pg['lr'] * p.grad.abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    if on_grad:
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
                    if try_on_wd:
                        if self.penatly_for_weight_decay:
                            p.grad.data += p.data.mul(weight_decay *
                                                      ((1 - penalty) / penalty))

    def apply_grad_only(self, from_grad=True):
        # This call does not flips the "skip one apply"
        # a following call to
        # `try_apply_wd_correction_before_step() must be done.
        self.apply(from_grad=from_grad, on_grad=True,
                   try_on_wd=False, ignore_skip_apply=True)

    def try_apply_wd_correction_before_step(self, from_grad=True):
        # This call also flips the "skip one apply".
        self.apply(from_grad=from_grad, on_grad=False,
                   try_on_wd=True, ignore_skip_apply=False)


# FIXME: keys are hardcoded from optimizers...
SGD_TYPE_TO_GAP_AWARE_CLASS = {
    'sgd1': GapAware,  # Pytorch
    # 'sgd2':   # TF # TODO
}

# optimizer, big_gamma=0.999, epsilon=1e-8, penatly_for_weight_decay=True


def get_sgd_gap_aware_cls(sgd_type: str) -> GapAware:
    gap_aware_cls = SGD_TYPE_TO_GAP_AWARE_CLASS.get(sgd_type, None)
    return gap_aware_cls
    # return gap_aware_cls(*args, **kw)


# class GAStats:
