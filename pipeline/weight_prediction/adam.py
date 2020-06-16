import torch
from .interface import WeightPredictor  # , FixFunction
import math


def get_adam_weight_predictor(pred_mem: str,
                              pred_type: str,
                              optimizer,
                              scheduler=None,
                              nag_with_predictor=False,
                              true_weights_storage=None) -> WeightPredictor:

    has_weight_decay = any(
        [pg['weight_decay'] != 0 for pg in optimizer.param_groups])

    # TODO: this is very minor, just use 1 thing *with* weight decay.
    if has_weight_decay:
        if pred_type == 'msnag':
            pred_cls = AdamClonedWeightPredictionWithWD
        elif pred_type == 'aggmsnag':
            pred_cls = AdamClonedWeightPredictionForAggregationWithWD
        else:
            raise NotImplementedError()
    else:
        if pred_type == 'msnag':
            pred_cls = AdamClonedWeightPrediction
        elif pred_type == 'aggmsnag':
            pred_cls = AdamClonedWeightPredictionForAggregationWithWD
        else:
            raise NotImplementedError()

    return pred_cls(optimizer,
                    fix_fn=None,
                    scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)


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
                for p in pg['params']:
                    state = self.optimizer.state[p]

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']
                    # NOTE: initial_step = step + 1

                    # Compute coefficient as sum of predictions.
                    # TODO: replace this with something more efficeint
                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):
                        if lr == 0:
                            continue

                        bias_correction1 = 1 - beta1**(step + staleness)
                        bias_correction2 = 1 - beta2**(step + staleness)

                        denom = (exp_avg_sq.sqrt() /
                                 math.sqrt(bias_correction2)).add_(eps)

                        step_size = lr / bias_correction1

                        p.data.addcdiv_(-step_size,
                                        exp_avg * (beta1**staleness), denom)

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
                    # TODO:  replace with something more efficient

                    exp_avg_hat = exp_avg

                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):

                        d_p = 0
                        if weight_decay != 0:
                            d_p = weight_decay * p.data

                        exp_avg_hat = exp_avg_hat * beta1 + (1 - beta1) * d_p
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


###########
# For aggregation:
###########


class AdamClonedWeightPredictionForAggregationWithWD(WeightPredictor):
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
                for p in pg['params']:
                    state = self.optimizer.state[p]

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']
                    weight_decay = pg['weight_decay']
                    # NOTE: initial_step = step + 1

                    # Compute coefficient as sum of predictions.
                    # TODO: replace this with something more efficeint

                    exp_avg_hat = exp_avg

                    for staleness, lr in zip(range(1, self.n_steps + 1),
                                             step_lrs):
                        if lr == 0:
                            continue

                        d_p = 0 if ((p.grad is None)
                                    or staleness > 1) else p.grad
                        # NOTE that loss is scaled by number of aggregation steps e.g mb=8 but we can have in p.grad sum of less than mb updates.
                        # We assume here nothing about next gradients (e.g no colliniarity, norm 0)

                        if weight_decay != 0:
                            d_p += weight_decay * p.data

                        exp_avg_hat = exp_avg_hat * beta1 + (1 - beta1) * d_p
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