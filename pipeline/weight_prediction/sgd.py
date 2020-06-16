import torch
from .interface import WeightPredictor, FixFunction
from .sgd_wd import SGDWDClonedWeightPrediction
import math

# """
# SGD: momentum fix coeff is the same for all parameters in group,
# as it is determined by LR and GAMMA
# """


class SGDRevertableLinearWeightPrediction(WeightPredictor):

    # FIXME: handle the error obtained from linear prediction (error < 1e-7)
    def __init__(self, *args, **kw):
        # TODO: work in interaction with "true_weights_storage" there is some hack to be done...
        raise NotImplementedError(
            "SGDRevertableLinearWeightPrediction not yet supported for pipeline"
        )
        super().__init__(*args, **kw)

    def forward(self):
        if not self.n_steps:
            return
        with torch.no_grad():
            self.buffered_fixes = [
                self.fix_fn(self, pg) for pg in self.optimizer.param_groups
            ]

            for pg, fix_fn_item in zip(self.optimizer.param_groups,
                                       self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.add_(-fix_fn_item,
                               self.optimizer.state[p]["momentum_buffer"])

    def revert(self):
        if not self.n_steps:
            return
        with torch.no_grad():
            for pg, fix_fn_item in zip(self.optimizer.param_groups,
                                       self.buffered_fixes):
                if fix_fn_item:
                    for p in pg['params']:
                        p.add_(
                            fix_fn_item,
                            self.optimizer.state[p]["momentum_buffer"])


class SGDClonedWeightPrediction(WeightPredictor):
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

        # Ugly hack, init momentum buffer to zeros before we start
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(
                    p)

    def forward(self):
        # TODO: maybe also skil when lr is 0.
        if not self.n_steps:
            return

        os_state = self.optimizer.state
        self.true_weights_storage.create_cloned_if_needed()
        self.true_weights_storage.record_change_mode("pred")
        pgs = self.optimizer.param_groups
        with torch.no_grad():
            buffered_fixes = [self.fix_fn(self, pg) for pg in pgs]

            for pg, fix_fn_item in zip(pgs, buffered_fixes):
                for p in pg['params']:
                    p.add_(-fix_fn_item, os_state[p]["momentum_buffer"])

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()


class SGD2MSNAG(FixFunction):
    """ 
    SGD version mention in Sutskever et al, also used in tensorflow.
    Mentioned as eq 10 Goyal et al.
    Fixed with MSNAG
     """
    def __call__(self, p: WeightPredictor, pg):
        gamma = pg['momentum']
        if p.n_steps == 1:
            return gamma
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
SGD_TYPE_TO_MSNAG_CLASS = {'sgd1': SGD1MSNAG, 'sgd2': SGD2MSNAG}


def get_sgd_weight_predictor(sgd_type: str,
                             pred_mem: str,
                             pred_type: str,
                             optimizer,
                             scheduler=None,
                             nag_with_predictor=False,
                             true_weights_storage=None) -> WeightPredictor:
    has_weight_decay = any(
        [pg['weight_decay'] != 0 for pg in optimizer.param_groups])
    if has_weight_decay:
        if sgd_type == 'sgd1':
            if pred_mem == 'clone':
                return SGDWDClonedWeightPrediction(
                    optimizer,
                    fix_fn=None,
                    scheduler=scheduler,
                    nag_with_predictor=nag_with_predictor,
                    true_weights_storage=true_weights_storage)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    else:
        fix_fn_cls = SGD_TYPE_TO_MSNAG_CLASS.get(sgd_type, None)
        fix_fn = fix_fn_cls()
        pred_cls = PRED_MEM_TO_CLASS.get(pred_mem, None)
        # pred_cls: WeightPredictor
        # fix_fn: FixFunction
        return pred_cls(optimizer,
                        fix_fn=fix_fn,
                        scheduler=scheduler,
                        nag_with_predictor=nag_with_predictor,
                        true_weights_storage=true_weights_storage)
