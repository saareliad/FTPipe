import torch
from .interface import WeightPredictor, FixFunction
import math


class AdamClonedWeightPrediction(WeightPredictor):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

        # Ugly hack, init momentum buffer to zeros before we start
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                self.optimizer.state[p]['exp_avg'] = torch.zeros_like(p)
                self.optimizer.state[p]['exp_avg_sq'] = torch.zeros_like(p)

    # FIXME

    def forward(self):
        # For each parameter
        # - Calculate momentum fix coefficient
        # - Do wieght prediction

        if not self.n_steps:
            return

        self.true_weights_storage.create_cloned_if_needed()
        self.true_weights_storage.record_change_mode("pred")

        with torch.no_grad():
            # init theta for clone

            for pg in self.param_groups:

                beta1, beta2 = pg['betas']
                eps = pg['eps']
                lr = pg['lr']

                # TODO: sched aware LR.

                for p in pg['params']:
                    state = self.optimizer.state[p]

                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']

                    # Compute coefficient as sum of predictions.
                    momentum_coeff = 0
                    for stalness in range(1, self.n_steps + 1):
                        bias_correction1 = 1 - beta1**(step + stalness)
                        # NOTE: bias_correction2 stays the same.
                        bias_correction2 = 1 - beta2**(step)
                        # denom = exp_avg_sq.sqrt().div(
                        #     (math.sqrt(bias_correction2)) + eps)
                        # NOTE: moved lr outside
                        step_size = 1 / bias_correction1
                        # NOTE: move exp_avg outside
                        momentum_coeff += (
                            step_size * ((math.sqrt(bias_correction2)) + eps))

                    p.add_(-lr * momentum_coeff, exp_avg / exp_avg_sq.sqrt())

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


def get_adam_weight_predictor(sgd_type: str,
                              pred_mem: str,
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
