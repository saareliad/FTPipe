import math
import warnings

import torch

from .adam import adam_init, get_adam_weight_predictor
from .interface import WeightPredictor


def get_adamw_weight_predictor(pred_mem: str,
                               pred_type: str,
                               optimizer,
                               scheduler=None,
                               nag_with_predictor=False,
                               true_weights_storage=None) -> WeightPredictor:
    has_weight_decay = any(
        [pg['weight_decay'] != 0 for pg in optimizer.param_groups])

    if has_weight_decay:
        if pred_type == 'msnag':
            pred_cls = AdamWClonedWeightPrediction
        elif pred_type == 'aggmsnag':
            pred_cls = AdamWClonedWeightPredictionForAggregation
        else:
            raise NotImplementedError()

    else:
        # Adam weight prediction its the exact same and more efficient so use it.
        warnings.warn(
            "using Adam weight prediciton instad of AdamW becuse weight decay is 0"
        )
        return get_adam_weight_predictor(
            pred_mem,
            pred_type,
            optimizer,
            scheduler=scheduler,
            nag_with_predictor=nag_with_predictor,
            true_weights_storage=true_weights_storage)

    return pred_cls(optimizer,
                    fix_fn=None,
                    scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)


class AdamWClonedWeightPrediction(WeightPredictor):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)
        adam_init(self.optimizer)

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
            for pg, step_lrs in zip(pgs, pg_step_lrs):

                beta1, beta2 = pg['betas']
                eps = pg['eps']
                weight_decay = pg['weight_decay']
                for p in pg['params']:
                    state = self.optimizer.state[p]

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']
                    # NOTE: initial_step = step + 1

                    # Compute coefficient as sum of predictions.
                    # TODO: replace with something more efficeint

                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):
                        if lr == 0:
                            continue

                        p.data.mul_(1 - lr * weight_decay)

                        bias_correction1 = 1 - beta1 ** (step + staleness)
                        bias_correction2 = 1 - beta2 ** (step + staleness)

                        denom = (exp_avg_sq.sqrt() /
                                 math.sqrt(bias_correction2)).add_(eps)

                        step_size = lr / bias_correction1

                        p.data.addcdiv_(exp_avg * (beta1 ** staleness),
                                        denom,
                                        value=-step_size)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()


###########
# For aggregation:
###########


class AdamWClonedWeightPredictionForAggregation(WeightPredictor):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)
        adam_init(self.optimizer)

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
            for pg, step_lrs in zip(pgs, pg_step_lrs):

                beta1, beta2 = pg['betas']
                eps = pg['eps']
                weight_decay = pg['weight_decay']
                for p in pg['params']:
                    state = self.optimizer.state[p]

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']
                    # NOTE: initial_step = step + 1

                    # Compute coefficient as sum of predictions.
                    # TODO: replace with something more efficeint

                    exp_avg_hat = exp_avg
                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):
                        if lr == 0:
                            continue

                        d_p = 0 if ((p.grad is None)
                                    or staleness > 1) else p.grad

                        p.data.mul_(1 - lr * weight_decay)

                        exp_avg_hat = exp_avg_hat * beta1 + (1 - beta1) * d_p
                        bias_correction1 = 1 - beta1 ** (step + staleness)
                        bias_correction2 = 1 - beta2 ** (step + staleness)

                        denom = (exp_avg_sq.sqrt() /
                                 math.sqrt(bias_correction2)).add_(eps)

                        step_size = lr / bias_correction1

                        p.data.addcdiv_(exp_avg_hat, denom, value=-step_size)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()
