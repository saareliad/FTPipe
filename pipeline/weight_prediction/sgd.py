import torch
from .interface import WeightPredictor, FixFunction
import math

# """
# SGD: momentum fix coeff is the same for all parameters in group,
# as it is determined by LR and GAMMA
# """


class SGDRevertableLinearWeightPrediction(WeightPredictor):

    # FIXME: handle the error obtained from linear prediction (error < 1e-7)
    def __init__(self, opt, params,
                 fix_fn, scheduler=None):

        super().__init__(opt, params,
                         fix_fn, scheduler=scheduler)

    def forward(self):
        with torch.no_grad():
            self.buffered_fixes = [self.fix_fn(
                self, pg) for pg in self.optimizer.param_groups]

            for pg, fix_fn_item in zip(self.optimizer.param_groups, self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.add_(-fix_fn_item,
                               self.optimizer.state[p]["momentum_buffer"].data)

    def revert(self):
        with torch.no_grad():
            for pg, fix_fn_item in zip(self.optimizer.param_groups, self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.add_(fix_fn_item,
                               self.optimizer.state[p]["momentum_buffer"].data)


class SGDClonedWeightPrediction(WeightPredictor):
    def __init__(self, opt, params,
                 momentum_fix_coeff_function, scheduler=None):

        super().__init__(opt, params,
                         momentum_fix_coeff_function, scheduler=scheduler)

    def forward(self):
        with torch.no_grad():
            # init theta for clone
            self.theta_buffer = [[p.data.clone() for p in pg['params']]
                                 for pg in self.optimizer.param_groups]

            self.buffered_fixes = [self.fix_fn(
                self, pg) for pg in self.optimizer.param_groups]
            for pg, fix_fn_item in zip(self.optimizer.param_groups, self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.add_(-fix_fn_item,
                               self.optimizer.state[p]["momentum_buffer"].data)

    def revert(self):
        with torch.no_grad():
            for pg, fix_fn_item, cloned in zip(self.optimizer.param_groups, self.buffered_fixes, self.theta_buffer):
                if fix_fn_item:
                    for p, bp in zip(pg['params'], cloned):
                        p.data = bp.data


class SGD1MSNAG(FixFunction):
    """ 
    SGD version mention in Sutskever et al, also used in tensorflow.
    Mentioned as eq 10 Goyal et al.
    Fixed with MSNAG
     """

    def __call__(self, p: WeightPredictor, pg):
        gamma = pg['momentum']
        # if p.n_steps == 1:
        #     return gamma
        return (gamma - math.pow(gamma, p.n_steps + 1)) / (1 - gamma)
        # return torch.tensor(gamma).pow_(d + 1).add_(- gamma).div_(gamma - 1)


class SGD2MSNAG(SGD1MSNAG):
    """ Pytorch SGD. Mentioned as eq 9 Goyal et al. """
    def __call__(self, p: WeightPredictor, pg):
        return pg['lr'] * super().__call__(p, pg)