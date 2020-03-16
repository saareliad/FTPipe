import torch
from .interface import WeightPredictor, FixFunction
import math


class SGDWDClonedWeightPrediction(WeightPredictor):
    """ Pytorch SGD. Mentioned as eq 9 Goyal et al.
        Used msnag to predict, including weight decay.

        # NOTE: when nag_with_predictor is on, 
        # it does what I developed as "weight decay aware nesterov", 
        # that is, using the weight decay to look forward, and not just the momentum.
        # (which most correctly implemented optimizers do anyway)
        # I elaborate on this on my notes.
     """
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


        os_state = self.optimizer.state
        self.true_weights_storage.create_cloned_if_needed()
        self.true_weights_storage.record_change_mode("pred")
        pgs = self.optimizer.param_groups
        os_state = self.optimizer.state
        with torch.no_grad():
            for pg in pgs:
                lr = pg['lr']
                if lr == 0 :
                    continue
                momentum = pg['momentum']
                weight_decay = pg['weight_decay']

                for p in pg['params']:
                    buff_hat = os_state[p]["momentum_buffer"].data
                    # NOTE: theta_hat = p.data
                    
                    # TODO: buff_hat requires extra memory, 
                    # there is probably a closed form way to compute this.

                    for staleness in range(1, self.n_steps + 1):
                        d_p = 0
                        if weight_decay != 0:
                            d_p = weight_decay * p.data
                            buff_hat = buff_hat * momentum + d_p
                            p.data.add_(-lr * buff_hat)

    def revert(self):
        if not self.n_steps:
            return
        self.true_weights_storage.restore_if_needed()
