import torch
from .interface import WeightPredictor, FixFunction
import math

# """
# SGD: momentum fix coeff is the same for all parameters in group,
# as it is determined by LR and GAMMA
# """


class SGDRevertableLinearWeightPrediction(WeightPredictor):

    # FIXME: handle the error obtained from linear prediction (error < 1e-7)
    def __init__(self, *args, **kw):
        raise NotImplementedError(
            "SGDRevertableLinearWeightPrediction not yet supported for pipeline")
        super().__init__(*args, **kw)

    def forward(self):
        if not self.n_steps:
            return
        with torch.no_grad():
            self.buffered_fixes = [self.fix_fn(
                self, pg) for pg in self.optimizer.param_groups]

            for pg, fix_fn_item in zip(self.optimizer.param_groups, self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.data.add_(-fix_fn_item,
                                    self.optimizer.state[p]["momentum_buffer"].data)

    def revert(self):
        if not self.n_steps:
            return
        with torch.no_grad():
            for pg, fix_fn_item in zip(self.optimizer.param_groups, self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.data.add_(fix_fn_item,
                                    self.optimizer.state[p]["momentum_buffer"].data)


class SGDClonedWeightPrediction(WeightPredictor):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

        # Ugly hack, init momentum buffer to zeros before we start
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(
                    p)

    def forward(self):
        if not self.n_steps:
            return
        with torch.no_grad():
            # init theta for clone
            # TODO: we are doing unneccery clone for batches without staleness.
            # e.g first batch in run_until_flush()
            # however when we NAG we predictor (current practice), the staleness is 1 instead of 0.

            self.true_weights_storage.create_cloned_if_needed()
            self.true_weights_storage.record_change_mode("pred")

            # self.theta_buffer = [[p.data.clone() for p in pg['params']]
            #                      for pg in self.optimizer.param_groups]
            pgs = self.optimizer.param_groups

            self.buffered_fixes = [self.fix_fn(self, pg) for pg in pgs]
            for pg, fix_fn_item in zip(pgs, self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.data.add_(-fix_fn_item,
                                    self.optimizer.state[p]["momentum_buffer"].data)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()

        # with torch.no_grad():
        #     for pg, fix_fn_item, cloned in zip(self.optimizer.param_groups, self.buffered_fixes, self.theta_buffer):
        #         if fix_fn_item:
        #             for p, bp in zip(pg['params'], cloned):
        #                 p.data = bp.data


class SGD2MSNAG(FixFunction):
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


class SGD1MSNAG(SGD2MSNAG):
    """ Pytorch SGD. Mentioned as eq 9 Goyal et al. """

    def __call__(self, p: WeightPredictor, pg):
        return pg['lr'] * super().__call__(p, pg)


PRED_MEM_TO_CLASS = {
    'clone': SGDClonedWeightPrediction,
    'calc': SGDRevertableLinearWeightPrediction
}

# FIXME: keys are hardcoded from optimizers...
SGD_TYPE_TO_MSNAG_CLASS = {
    'sgd1': SGD1MSNAG,
    'sgd2': SGD2MSNAG
}


def get_sgd_weight_predictor(sgd_type: str, pred_mem: str,
                             optimizer, scheduler=None, 
                             nag_with_predictor=False, 
                             true_weights_storage=None) -> WeightPredictor:
    fix_fn_cls = SGD_TYPE_TO_MSNAG_CLASS.get(sgd_type, None)
    fix_fn = fix_fn_cls()
    pred_cls = PRED_MEM_TO_CLASS.get(pred_mem, None)
    # pred_cls: WeightPredictor
    # fix_fn: FixFunction
    return pred_cls(optimizer, fix_fn, scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)
