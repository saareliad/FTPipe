import torch
from .interface import GapAwareBase
import abc
from itertools import chain

# TODO: check that the step count is indeed OK (and ahead by 1)


def opt_params_iter(optimizer):
    return chain(*[pg['params'] for pg in optimizer.param_groups])


class AdamGapAware(GapAwareBase):
    """ Gap aware for ADAM optimizer """
    def __init__(self,
                 optimizer,
                 big_gamma=0.999,
                 epsilon=1e-8,
                 from_grad=True):
        """ Apply Gap Aware on computed gradients """
        super().__init__(optimizer)

        # State initialization
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                state = self.optimizer.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)
                    state['step'] = 0

        #     # TODO: sched aware LR.

    def apply_from_grad(self):
        """ Calculate gap aware from gradient. Requires knowing the exact gap """
        raise NotImplementedError()

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
                    exp_avg_sq = opt_state[p]['exp_avg_sq']

                    # FIXME: remove assert after this works.
                    step_count = opt_state[p]['step'] + 1
                    assert step_count == self.step_count

                    bias_correction2 = 1 - beta2**(step_count)
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((exp_avg_sq.data / bias_correction2) ** 0.5) + eps)

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

        raise NotImplementedError()

    def apply_on_theta(self, real_theta):

        opt_state = self.optimizer.state

        with torch.no_grad():
            for pg, rpg in zip(self.optimizer.param_groups, real_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']

                for p, rp in zip(pg['params'], rpg):
                    exp_avg_sq = opt_state[p]['exp_avg_sq']
                    # FIXME: remove assert after this works.
                    step_count = opt_state[p]['step'] + 1
                    assert step_count == self.step_count

                    bias_correction2 = 1 - beta2**(step_count)
                    # if p.grad is None:
                    #     continue
                    # calculate C coefficient per-element
                    # Note: can remove the "data". but whatever.
                    avg_steps_needed = max_lr * \
                        (((exp_avg_sq.data / bias_correction2) ** 0.5) + eps)

                    gap = (p - rp).abs()
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
                    p.grad.data += rp.data.mul(weight_decay *
                                               ((1 - penalty) / penalty))

    def update_running_avg(self):
        """in case there is some running avg to update"""
        pass

    # TODO: after getting step from optimizer works.
    # def inc_step_count(self):
    #     pass


def get_adam_gap_aware_cls() -> AdamGapAware:
    gap_aware_cls = AdamGapAware
    return gap_aware_cls
