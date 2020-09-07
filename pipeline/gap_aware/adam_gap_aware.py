import math
from itertools import chain

import numpy as np
import torch

from .interface import GapAwareBase


def gap_aware_adam_init(optimizer):
    # Ugly hack, init momentum buffer to zeros before we start
    # State initialization
    for pg in optimizer.param_groups:
        for p in pg['params']:
            state = optimizer.state[p]
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format)
                state['step'] = 0
                # Exponential moving average of squared step values
                state['exp_step_avg_sq'] = torch.zeros_like(
                    p.data, memory_format=torch.preserve_format)
                # NOTE: amsgrad is not supported.


# TODO: check that the step count is indeed OK (and ahead by 1)
# TODO: record and return total gap.


def opt_params_iter(optimizer):
    return chain(*[pg['params'] for pg in optimizer.param_groups])


def adam_gap1(beta1, beta2, eps, eta, gt, m, v):
    tmp0 = (1 - beta1)
    sq_tmp0 = math.sqrt(tmp0)
    tmp1 = tmp0 * eps
    tmp2 = m * (eta * beta1)
    nom1 = tmp2
    dnom1 = tmp1 + torch.sqrt(v) * sq_tmp0

    e1 = nom1 / dnom1  # todo can save memory by out to denom1
    # todo maybe can save mem by multiple pointers.

    nom2_1 = tmp2
    nom2_2 = gt * (tmp0 * eta)
    nom2 = nom2_1 + nom2_2

    tmp4 = tmp0 / math.sqrt(1 - beta2)
    dnom2 = (beta2 * v + (gt ** 2) * (1 - beta2)).sqrt_().mul_(tmp4).add_(tmp1)

    e2 = nom2 / dnom2  # todo can save memory by inplace to either nom2 or dnom2

    return e1 - e2


class AdamGapAware(GapAwareBase):
    """ Gap aware for ADAM optimizer """

    def __init__(self, optimizer, from_grad=False):  # FIXME:?
        """ Apply Gap Aware on computed gradients """
        super().__init__(optimizer)

        # NOTE: the goal of the running avg is to estimate the step size we take
        gap_aware_adam_init(optimizer)
        #     # TODO: sched aware LR.

    def apply_from_grad(self):
        """ Calculate gap aware from gradient. Requires knowing the exact gap.
            Requires: weight prediction.
            # TODO: to handle some micro batches which do not have grad, we can aggregate updates for stale gradients with extra memory cost 
            # (its not THAT much)
        """
        # -\beta_{1}*\eta*m/(-\beta_{1}*\epsilon + \epsilon + sqrt(v)*(-\beta_{1}/sqrt(1 - \beta_{1}) + 1/sqrt(1 - \beta_{1}))) + theta

        opt_state = self.optimizer.state
        with torch.no_grad():
            for pg in self.optimizer.param_groups:

                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']
                eta = pg['lr']

                if weight_decay != 0:
                    raise NotImplementedError()

                for p in pg['params']:
                    avg_steps_needed = (
                                               (opt_state[p]['exp_step_avg_sq']) ** 0.5) + eps

                    m = opt_state[p]['exp_avg']
                    v = opt_state[p]['exp_avg_sq']
                    gt = p.grad
                    gap = adam_gap1(beta1, beta2, eps, eta, gt, m, v).abs_()

                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # TODO:
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # p.grad.data += p.data.mul(weight_decay *
                    #                           ((1 - penalty) / penalty))

    def apply_on_stashed(self, stashed_theta):
        """ True weights are loaded into the model, and given a stashed theta """
        opt_state = self.optimizer.state

        with torch.no_grad():
            for pg, spg in zip(self.optimizer.param_groups, stashed_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']

                for p, sp in zip(pg['params'], spg):
                    # step_count = opt_state[p]['step'] + 1
                    # bias_correction2 = 1 - beta2**(step_count)

                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    # TODO: use max_lr somwhow, instead of doing it in optimizer
                    avg_steps_needed = (
                                               (opt_state[p]['exp_step_avg_sq'].data) ** 0.5) + eps

                    gap = (p - sp).abs()
                    # pg['lr'] * p.grad.abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # NOTE: the memory hack below also worked for SGD.
                    # HACK: we know that adam does
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

    def apply_on_theta(self, real_theta):

        opt_state = self.optimizer.state

        penatly_arr = []

        with torch.no_grad():
            for pg, rpg in zip(self.optimizer.param_groups, real_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']

                for p, rp in zip(pg['params'], rpg):
                    # step_count = opt_state[p]['step'] + 1
                    # bias_correction2 = 1 - beta2**(step_count)

                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.

                    # TODO: use max_lr somwhow, instead of doing it in optimizer
                    avg_steps_needed = (
                                               (opt_state[p]['exp_step_avg_sq'].data) ** 0.5) + eps

                    gap = (p - rp).abs()
                    # pg['lr'] * p.grad.abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    penatly_arr.append(torch.mean(penalty).item())

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # NOTE: the memory hack below also worked for SGD.
                    # HACK: we know that adam does
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

        print("mean_penaltly", np.mean(penatly_arr))

    def update_running_stats(self):
        pass


def get_adam_gap_aware_cls() -> AdamGapAware:
    gap_aware_cls = AdamGapAware
    return gap_aware_cls
