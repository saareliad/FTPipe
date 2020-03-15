import torch
from .interface import WeightPredictor, FixFunction
import math


class AdamClonedWeightPrediction(WeightPredictor):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

        # Ugly hack, init momentum buffer to zeros before we start
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
                # NOTE: amsgrad is not supported.

    def forward(self):
        if not self.n_steps:
            return

        self.true_weights_storage.create_cloned_if_needed()
        self.true_weights_storage.record_change_mode("pred")
        pgs = self.optimizer.param_groups

        # get LRs from scheduler (sched-aware)
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
                # TODO: sched aware LR.

                for p in pg['params']:
                    state = self.optimizer.state[p]

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']
                    initial_step = step + 1  # HACK: add +1 so it won't be 0.

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


class AdamMSNAG(FixFunction):
    """ 
    ADAM msnag
    """

    def __call__(self, p: WeightPredictor, pg):
        raise NotImplementedError(
            "its intentionally empty, Unlike SGD I currently do the fix inline"
        )


def get_adam_weight_predictor(pred_mem: str,
                              optimizer,
                              scheduler=None,
                              nag_with_predictor=False,
                              true_weights_storage=None) -> WeightPredictor:
    # fix_fn_cls = SGD_TYPE_TO_MSNAG_CLASS.get(sgd_type, None)
    # fix_fn = fix_fn_cls()
    # pred_cls = PRED_MEM_TO_CLASS.get(pred_mem, None)
    fix_fn = AdamMSNAG()
    pred_cls = AdamClonedWeightPrediction
    # pred_cls: WeightPredictor
    # fix_fn: FixFunction
    return pred_cls(optimizer,
                    fix_fn,
                    scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)
