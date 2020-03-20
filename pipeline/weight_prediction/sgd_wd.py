import torch
from .interface import WeightPredictor, FixFunction
import math
from .sympy_pred_optimizers import auto_lambdify, WDSympySGD


class SGDWDClonedWeightPrediction(WeightPredictor):
    """ Pytorch SGD. Mentioned as eq 9 Goyal et al.
        Used msnag to predict, including weight decay.
     """
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

        # Ugly hack, init momentum buffer to zeros before we start
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(
                    p)

        MAX_ALLOWEDD_STALENESS = 8  # TODO: should be a variable but I want to keep things simple.

        # Automaticallty create functions to compute coeffs given staleness
        res, _ = auto_lambdify(MAX_ALLOWEDD_STALENESS,
                               WDSympySGD,
                               simplify=True)
        self.res = res

    def setup(self, n_steps):
        # Overriding this function, we handle nag_with_predictor in forward().
        self.n_steps = n_steps

    def forward(self):
        if self.n_steps == 0 and self.nag_with_predictor:
            self.n_steps = 1
            # add without weight decay.
            os_state = self.optimizer.state
            self.true_weights_storage.create_cloned_if_needed()
            self.true_weights_storage.record_change_mode("pred")
            with torch.no_grad():
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                    if lr == 0:
                        continue
                    momentum = pg['momentum']
                    for p in pg['params']:
                        p.data.add_(-lr * momentum,
                                    os_state[p]["momentum_buffer"].data)

        if not self.n_steps:
            return

        # Extract coefficients and symbols
        res = self.res[self.n_steps]
        res_v = res['v']
        res_theta = res['theta']
        f_v = res_v['f']
        f_theta = res_theta['f']
        fs_v = res_v["free_symbols"]
        fs_theta = res_theta["free_symbols"]

        os_state = self.optimizer.state
        self.true_weights_storage.create_cloned_if_needed()
        self.true_weights_storage.record_change_mode("pred")
        pgs = self.optimizer.param_groups
        with torch.no_grad():
            for pg in pgs:
                # if lr == 0:
                #    continue
                # dict to map params to symbols
                # TODO: can change and init to avoid creating this here...

                d = {
                    '\\eta': pg['lr'],
                    '\\gamma': pg['momentum'],
                    '\\lambda': pg['weight_decay']
                }

                coeff_v = f_v(*[d[a] for a in fs_v])
                coeff_theta = f_theta(*[d[a] for a in fs_theta])
                for p in pg['params']:
                    p.data.mul_(coeff_theta).add_(
                        coeff_v, os_state[p]["momentum_buffer"].data)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()
