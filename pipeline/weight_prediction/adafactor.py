import math

import torch

from .cow_dict import CowDict
from .interface import WeightPredictor


def get_adafactor_weight_predictor(
        pred_mem: str,
        pred_type: str,
        optimizer,
        scheduler=None,
        nag_with_predictor=False,
        true_weights_storage=None) -> WeightPredictor:
    has_weight_decay = any(
        [pg['weight_decay'] != 0 for pg in optimizer.param_groups])

    if has_weight_decay:
        pass

    if pred_type == 'msnag':
        raise NotImplementedError()
        # pred_cls = AdaFactorWClonedWeightPrediction
    elif pred_type == 'aggmsnag':
        pred_cls = AdaFactorWClonedWeightPredictionForAggregation
    else:
        raise NotImplementedError()

    return pred_cls(optimizer,
                    fix_fn=None,
                    scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)


def adafactor_init(optimizer):
    # Ugly hack, init momentum buffer to zeros before we start
    # State initialization
    for pg in optimizer.param_groups:
        for p in pg['params']:
            state = optimizer.state[p]

            grad = p
            grad_shape = grad.shape

            factored, use_first_moment = optimizer._get_options(pg, grad_shape)
            # State Initialization
            if len(state) == 0:
                state['step'] = 0

                if use_first_moment:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(
                        grad_shape[:-1]).to(grad)
                    state['exp_avg_sq_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:]).to(grad)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                state['RMS'] = 0


class AdaFactorWClonedWeightPredictionForAggregation(WeightPredictor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        from optimizers.adafactor import Adafactor
        self.optimizer: Adafactor

        adafactor_init(self.optimizer)

    def forward(self):
        if not self.n_steps:
            return

        self.true_weights_storage.create_cloned_if_needed()
        self.true_weights_storage.record_change_mode("pred")
        pgs = self.optimizer.param_groups

        # get LRs from scheduler (sched-aware)
        # NOTE: self.scheduler is sched_aware...
        if self.scheduler is not None:
            step_lrs = self.scheduler.get_next(self.n_steps)
            pg_step_lrs = [[slr[i] for slr in step_lrs]
                           for i in range(len(pgs))]

        else:
            pg_step_lrs = [[pg['lr']] * self.n_steps for pg in pgs]

        with torch.no_grad():
            # for pg, step_lrs in zip(pgs, pg_step_lrs):

            #####################################

            for group, step_lrs in zip(pgs, pg_step_lrs):

                group = CowDict(group)

                for p in group['params']:
                    if p.grad is None:
                        grad = None
                    else:
                        grad = p.grad.data
                        if grad.dtype in {torch.float16, torch.bfloat16}:
                            grad = grad.float()

                    state = self.optimizer.state[p]
                    state = CowDict(state)
                    grad_shape = grad.shape if grad is not None else p.shape
                    factored, use_first_moment = self.optimizer._get_options(
                        group, grad_shape)
                    # State Initialization

                    assert len(state) > 0

                    # Pre-read these.
                    # The inplace ops are replaced by out of place ops so the actual value will not change!
                    if factored:
                        exp_avg_sq_row = state['exp_avg_sq_row']
                        exp_avg_sq_col = state['exp_avg_sq_col']
                    else:
                        exp_avg_sq = state['exp_avg_sq']

                    if use_first_moment:
                        exp_avg = state['exp_avg']

                    # TODO: fp16
                    # t = grad if grad is not  None else p
                    # if use_first_moment:
                    #     state['exp_avg'] = state['exp_avg'].to(grad)
                    # if factored:
                    #     state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(grad)
                    #     state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(grad)
                    # else:
                    #     state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):

                        p_data_fp32 = p.data
                        if p.data.dtype in {torch.float16, torch.bfloat16}:
                            p_data_fp32 = p_data_fp32.float()

                        # This is handled by copy in write! awssome!
                        state['step'] += 1
                        state['RMS'] = self.optimizer._rms(p_data_fp32)
                        group['lr'] = self.optimizer._get_lr(group, state)

                        beta2t = 1.0 - math.pow(state['step'],
                                                group['decay_rate'])
                        if grad is None:
                            update = torch.full_like(
                                p.data,
                                fill_value=group['eps'][0],
                                memory_format=torch.preserve_format)
                        else:
                            update = (grad ** 2) + group['eps'][0]
                        if factored:
                            # exp_avg_sq_row = state['exp_avg_sq_row']
                            # exp_avg_sq_col = state['exp_avg_sq_col']

                            # TODO: Following 2 lines may be redundent
                            exp_avg_sq_row = exp_avg_sq_row.mul(beta2t).add_(
                                update.mean(dim=-1), alpha=1.0 - beta2t)
                            exp_avg_sq_col = exp_avg_sq_col.mul(beta2t).add_(
                                update.mean(dim=-2), alpha=1.0 - beta2t)

                            # Approximation of exponential moving average of square of gradient
                            update = self.optimizer._approx_sq_grad(
                                exp_avg_sq_row, exp_avg_sq_col)
                            update.mul_(grad)
                        else:
                            # exp_avg_sq = state['exp_avg_sq']

                            exp_avg_sq = exp_avg_sq.mul(beta2t).add_(
                                update, alpha=1.0 - beta2t)
                            update = exp_avg_sq.rsqrt().mul_(grad)

                        update.div_((self.optimizer._rms(update) /
                                     group['clip_threshold']).clamp_(min=1.0))
                        update.mul_(group['lr'])

                        if use_first_moment:
                            # exp_avg = state['exp_avg']
                            exp_avg = exp_avg.mul(group['beta1']).add_(
                                update, alpha=1 - group['beta1'])
                            update = exp_avg

                        if group['weight_decay'] != 0:
                            p_data_fp32.add_(p_data_fp32,
                                             alpha=-group['weight_decay'] *
                                                   group['lr'])

                        p_data_fp32.add_(-update)

                        if p.data.dtype in {torch.float16, torch.bfloat16}:
                            p.data.copy_(p_data_fp32)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()

###########
# For aggregation:
###########

#
# class AdamWClonedWeightPredictionForAggregation(WeightPredictor):
#     def __init__(self, *args, **kw):
#
#         super().__init__(*args, **kw)
#         adam_init(self.optimizer)
#
#     def forward(self):
#         if not self.n_steps:
#             return
#
#         self.true_weights_storage.create_cloned_if_needed()
#         self.true_weights_storage.record_change_mode("pred")
#         pgs = self.optimizer.param_groups
#
#         # get LRs from scheduler (sched-aware)
#         # NOTE: self.scheduler is sched_aware...
#         if self.scheduler is not None:
#             step_lrs = self.scheduler.get_next(self.n_steps)
#             pg_step_lrs = [[slr[i] for slr in step_lrs]
#                            for i in range(len(pgs))]
#
#         else:
#             pg_step_lrs = [[pg['lr']] * self.n_steps for pg in pgs]
#
#         with torch.no_grad():
#             for pg, step_lrs in zip(pgs, pg_step_lrs):
#
#                 beta1, beta2 = pg['betas']
#                 eps = pg['eps']
#                 weight_decay = pg['weight_decay']
#                 for p in pg['params']:
#                     state = self.optimizer.state[p]
#
#                     exp_avg = state['exp_avg']
#                     exp_avg_sq = state['exp_avg_sq']
#                     step = state['step']
#                     # NOTE: initial_step = step + 1
#
#                     # Compute coefficient as sum of predictions.
#                     # TODO: replace with something more efficeint
#
#                     exp_avg_hat = exp_avg
#                     for staleness, lr in zip(range(1, self.n_steps + 1),
#                                              step_lrs):
#                         if lr == 0:
#                             continue
#
#                         d_p = 0 if ((p.grad is None)
#                                     or staleness > 1) else p.grad
#
#                         p.data.mul_(1 - lr * weight_decay)
#
#                         exp_avg_hat = exp_avg_hat * beta1 + (1 - beta1) * d_p
#                         bias_correction1 = 1 - beta1**(step + staleness)
#                         bias_correction2 = 1 - beta2**(step + staleness)
#
#                         denom = (exp_avg_sq.sqrt() /
#                                  math.sqrt(bias_correction2)).add_(eps)
#
#                         step_size = lr / bias_correction1
#
#                         p.data.addcdiv_(exp_avg_hat, denom, value=-step_size)
#
#     def revert(self):
#         if not self.n_steps:
#             return
#         self.true_weights_storage.restore_if_needed()
