import torch
from .interface import GapAwareBase
from itertools import chain
from .sgd_gap_aware import init_running_avg_step
from ..weight_prediction.adam import adam_init
import numpy as np

# TODO: check that the step count is indeed OK (and ahead by 1)
# TODO: record and return total gap.


def opt_params_iter(optimizer):
    return chain(*[pg['params'] for pg in optimizer.param_groups])


class AdamGapAware(GapAwareBase):
    """ Gap aware for ADAM optimizer """
    def __init__(self, optimizer, from_grad=False):  # FIXME:?
        """ Apply Gap Aware on computed gradients """
        super().__init__(optimizer)

        self.running_avg_step = init_running_avg_step(optimizer)
        adam_init(optimizer)
        #     # TODO: sched aware LR.

    def apply_from_grad(self):
        """ Calculate gap aware from gradient. Requires knowing the exact gap """
        raise NotImplementedError()

    def apply_on_stashed(self, stashed_theta):
        """ True weights are loaded into the model, and given a stashed theta """
        opt_state = self.optimizer.state
        ra = self.running_avg_step

        with torch.no_grad():
            for pg, spg in zip(self.optimizer.param_groups, stashed_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']

                for p, sp in zip(pg['params'], spg):

                    step_count = opt_state[p]['step'] + 1

                    bias_correction2 = 1 - beta2**(step_count)
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((ra[id(p)].data / bias_correction2) ** 0.5) + eps)

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
        ra = self.running_avg_step

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
                    step_count = opt_state[p]['step'] + 1

                    bias_correction2 = 1 - beta2**(step_count)
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((ra[id(p)].data / bias_correction2) ** 0.5) + eps)

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
        """
        Update the exponential step running average
        Requires: that we got some grad.
        """
        # NOTE: same as adamw_gap_aware
        opt_s = self.optimizer.state
        ra = self.running_avg_step

        with torch.no_grad():
            for pg in self.optimizer.param_groups:
                beta1, beta2 = pg['betas']

                if beta1 != 0:
                    for p in pg['params']:
                        ra[id(p)].data = beta2 * ra[id(p)].data + \
                            (1 - beta2) * \
                            (opt_s[p]["exp_avg"].data ** 2)
                else:
                    for p in pg['params']:
                        ra[id(p)].data = opt_s[p]['exp_avg_sq'].data


def get_adam_gap_aware_cls() -> AdamGapAware:
    gap_aware_cls = AdamGapAware
    return gap_aware_cls
