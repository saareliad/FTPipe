import torch
from .interface import WeightPredictor, FixFunction
import math


def adam_init(optimizer):
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
                # NOTE: amsgrad is not supported.

class AdamClonedWeightPrediction(WeightPredictor):
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

                    momentum_coeff = 0

                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):

                        bias_correction1 = 1 - beta1**(step + staleness)
                        bias_correction2 = 1 - beta2**(step + staleness)

                        denom = (exp_avg_sq.sqrt() /
                                 math.sqrt(bias_correction2)).add_(eps)

                        step_size = lr / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg *
                                        (beta1**staleness), denom)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()


class AdamClonedWeightPredictionWithWD(WeightPredictor):
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
                    

                    momentum_coeff = 0
                    exp_avg_hat = exp_avg

                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):
                        d_p = 0
                        if weight_decay !=0:
                            d_p = weight_decay * p.data

                        exp_avg_hat = exp_avg * beta1 + (1-beta1) * d_p
                        bias_correction1 = 1 - beta1**(step + staleness)
                        bias_correction2 = 1 - beta2**(step + staleness)

                        denom = (exp_avg_sq.sqrt() /
                                 math.sqrt(bias_correction2)).add_(eps)

                        step_size = lr / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg_hat, denom)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()


def get_adam_weight_predictor(pred_mem: str,
                              optimizer,
                              scheduler=None,
                              nag_with_predictor=False,
                              true_weights_storage=None) -> WeightPredictor:
    

    has_weight_decay = any(
        [pg['weight_decay'] != 0 for pg in optimizer.param_groups])

    if has_weight_decay:
        pred_cls = AdamClonedWeightPredictionWithWD
    else:
        pred_cls = AdamClonedWeightPrediction


    return pred_cls(optimizer,
                    fix_fn=None,
                    scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)
