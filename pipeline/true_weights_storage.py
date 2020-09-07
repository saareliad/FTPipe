import torch


class TrueWeightsStorage:
    """
    NOTE: in case of multiple restores, we take a "copy on write" approach.
        Frist restore: no clone -> copy pointers.
        Second restore: clone the true weights ("pop" the previous ones-to the model)
        This is handled by `self.restored_true_weights_to_the_model`.

    """

    def __init__(self, optimizer):
        self.true_weights = None
        self.true_weights_exist = False
        self.optimizer = optimizer
        self.change_mode = None
        self.restored_true_weights_to_the_model = False

    def record_change_mode(self, mode):
        if self.change_mode is None:
            self.change_mode = mode
        elif mode:
            self.change_mode += f" -> {mode}"

    def get_true_weights(self):
        true_weights = self.true_weights
        # HACK
        if true_weights is None:
            true_weights = self._return_current_weights()
        return true_weights

    # def set_true_weights_buffer(self, buff):
    #     self.true_weights = buff
    #     self.true_weights_exist = True

    def create_cloned_if_needed(self):
        if (not self.true_weights_exist) or self.restored_true_weights_to_the_model:
            self.true_weights = self._create_current_cloned_buff()
            self.true_weights_exist = True

            # Flip
            if self.restored_true_weights_to_the_model:
                self.restored_true_weights_to_the_model = False

    def restore_if_needed(self):
        if self.true_weights_exist and self.change_mode:
            self._restore_from_buff(self.true_weights)
            self.change_mode = None

            # A naive solution here is to pop,
            # A smarter solution is to clone on the second time "copy on write"
            self.restored_true_weights_to_the_model = True

    def check_restore_if_needed(self, check=True):
        """ Function to check restore_if_needed calls """
        if check:
            if self.true_weights_exist and not self.change_mode:
                print(
                    "-W- will not restore true weights. no change is recorded. Consider removing for efficiency")
        self.restore_if_needed()

    def reset_on_step(self):
        self.true_weights = None
        self.true_weights_exist = False
        self.change_mode = None
        # NOTE: this is not really neccasarry
        self.restored_true_weights_to_the_model = False

    def _restore_from_buff(self, buff):
        """load tensor.data from saved buff. Gradients stays the same."""
        with torch.no_grad():
            for pg, cloned in zip(self.optimizer.param_groups, buff):
                for p, bp in zip(pg['params'], cloned):
                    p.data = bp.detach()

    def _create_current_cloned_buff(self):
        buff = [[p.detach().clone() for p in pg['params']]
                for pg in self.optimizer.param_groups]
        return buff

    def _return_current_weights(self):
        return [[p for p in pg['params']]
                for pg in self.optimizer.param_groups]
