import torch
from .interface import GapAwareBase
from itertools import chain
from .adam_gap_aware import gap_aware_adam_init
# TODO: check that the step count is indeed OK (and ahead by 1)


def opt_params_iter(optimizer):
    return chain(*[pg['params'] for pg in optimizer.param_groups])


class AdamWGapAware(GapAwareBase):
    """ Gap aware for ADAMW optimizer
    ADAMW
    https://arxiv.org/pdf/1711.05101.pdf

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    # NOTE: it will also effect our weight prediction!!
    # Instead we want to decay the weights in a manner that doesn't interact
    # with the m/v parameters. This is equivalent to adding the square
    # of the weights to the loss with plain (non-momentum) SGD.


    Based on pytorch ADAMW implementation
    https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW
    NOTE: were straight at the beggining we do
               # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])


    NOTE: not all implemenetations are equivalent, e.g  transformers AdamW version
        does the update at the end (like the paper).
    I found that doing at the beggining is little better for weight prediction for SGD so I choose pytorch.

    https://github.com/huggingface/transformers/blob/19a63d8245f4ce95595a8be657eb669d6491cdf8/src/transformers/optimization.py#L96
        NOTE: they do it at the end, like the paper.
        if group["weight_decay"] > 0.0:
            p.data.add_(-group["lr"] * group["weight_decay"], p.data)

    """
    def __init__(self,
                 optimizer,
                 big_gamma=0.999,
                 epsilon=1e-8,
                 from_grad=True):
        """ Apply Gap Aware on computed gradients """
        super().__init__(optimizer)
        gap_aware_adam_init(optimizer)

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
                lr = pg['lr']
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']

                for p, sp in zip(pg['params'], spg):

                    # step_count = opt_state[p]['step'] + 1
                    # bias_correction2 = 1 - beta2**(step_count)

                    # calculate C coefficient per-element
                    # NOTE: can remove the "data". but whatever.
                    # TODO: use max_lr somwhow, instead of doing it in optimizer
                    avg_steps_needed = (opt_state[p]['exp_step_avg_sq'].data ** 0.5) + eps

                    gap = (p - sp).abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # NOTE: the memory hack below also worked for SGD.
                    # HACK: we know that adamW does
                    # p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    # p *= (1-lr*wd)  <==>  p -= p*lr*wd
                    # and we want:
                    # p *= (1-lr*wd/penalty)   <==> p -= p*lr*wd/penalty
                    # so we solve:
                    # x + z - p*lr*wd = x - (p*lr*wd / penalty)
                    # giving:
                    # z = -p*wd*lr ((1/penalty) - 1) = -p*wd*lr ((1 - penalty) / penalty)
                    # so we do
                    #   p += z
                    # equivalent to:
                    #   p.data.mul_(1+z/p)

                    # NOTE: we apply the weight decay on the real parameter weight, p.
                    p.data.mul_(1 - weight_decay * lr *
                                ((1 - penalty) / penalty))

        raise NotImplementedError()

    def apply_on_theta(self, real_theta):

        opt_state = self.optimizer.state

        with torch.no_grad():
            for pg, rpg in zip(self.optimizer.param_groups, real_theta):
                max_lr = pg[GapAwareBase.MAX_LR_NAME]
                lr = pg['lr']
                if max_lr <= 0:
                    continue
                weight_decay = pg['weight_decay']
                beta1, beta2 = pg['betas']
                eps = pg['eps']

                for p, rp in zip(pg['params'], rpg):

                    # step_count = opt_state[p]['step'] + 1
                    # bias_correction2 = 1 - beta2**(step_count)

                    # calculate C coefficient per-element
                    # NOTE: can remove the "data". but whatever.
                    # TODO: use max_lr somwhow, instead of doing it in optimizer
                    avg_steps_needed = (opt_state[p]['exp_step_avg_sq'].data ** 0.5) + eps

                    gap = (p - rp).abs()
                    # pg['lr'] * p.grad.abs()

                    # calculate the gap per-element
                    penalty = 1 + (gap / avg_steps_needed)

                    # Apply penalty to gradient
                    p.grad.data /= penalty
                    # Apply penalty to weight decay (as it will be part of the gradient)
                    # NOTE: the memory hack below also worked for SGD.
                    # HACK: we know that adamW does
                    # p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    # p *= (1-lr*wd)  <==>  p -= p*lr*wd
                    # and we want:
                    # p *= (1-lr*wd/penalty)   <==> p -= p*lr*wd/penalty
                    # so we solve:
                    # x + z - p*lr*wd = x - (p*lr*wd / penalty)
                    # giving:
                    # z = -p*wd*lr ((1/penalty) - 1) = -p*wd*lr ((1 - penalty) / penalty)
                    # so we do
                    #   p += z
                    # equivalent to:
                    #   p.data.mul_(1+z/p)

                    # NOTE: we apply the weight decay on the real parameter weight, rp.

                    rp.data.mul_(1 - weight_decay * lr *
                                 ((1 - penalty) / penalty))

    def update_running_stats(self):
        pass


# TODO: add to adam...
def get_adamw_gap_aware_cls() -> AdamWGapAware:
    gap_aware_cls = AdamWGapAware
    return gap_aware_cls
