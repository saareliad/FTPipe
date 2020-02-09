import torch


class TrueWeightsStorage:
    def __init__(self, optimizer):
        self.true_weights = None
        self.true_weights_exist = False
        self.optimizer = optimizer
        self.change_mode = None

    #     self.model_has_true = True

    # def set_mode(self, mode):
    #     self.model_has_true
    #     self.mode = mode

    def record_change_mode(self, mode):
        if self.change_mode is None:
            self.change_mode = mode
        elif mode:
            self.change_mode += f" -> {mode}"

    def get_true_weights(self):
        return self.true_weights

    def set_true_weights_buffer(self, buff):
        self.true_weights = buff
        self.true_weights_exist = True

    def create_cloned_if_needed(self):
        if (not self.true_weights_exist):
            buff = self._create_current_cloned_buff()
            self.set_true_weights_buffer(buff)

    def restore_if_needed(self):
        if self.true_weights_exist and self.change_mode:
            self._restore_from_buff(self.true_weights)
            self.change_mode = None

    def check_restore_if_needed(self, check=True):
        """ Function to check restore_if_needed calls """
        if check:
            if self.true_weights_exist and not self.change_mode:
                print("-W- will not restore true weights. no change is recorded. Consider removing for efficiency") 
        self.restore_if_needed()

    def reset_on_step(self):
        self.true_weights = None
        self.true_weights_exist = False
        self.change_mode = None

    def _restore_from_buff(self, buff):
        with torch.no_grad():
            for pg, cloned in zip(self.optimizer.param_groups, buff):
                for p, bp in zip(pg['params'], cloned):
                    p.data = bp.data

    def _create_current_cloned_buff(self):
        with torch.no_grad():
            buff = [[p.data.clone() for p in pg['params']]
                    for pg in self.optimizer.param_groups]
        return buff
