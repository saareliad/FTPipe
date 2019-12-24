import torch
from collections import OrderedDict


class WeightStasher:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.theta_buffer = OrderedDict()

    def create_current_cloned_buff(self):
        with torch.no_grad():
            buff = [[p.data.clone() for p in pg['params']]
                    for pg in self.optimizer.param_groups]
        return buff

    def current_is_last(self, batch_index):
        return next(reversed(self.theta_buffer.keys())) == batch_index

    def stash_if_current_is_last(self, batch_index):
        if self.current_is_last(batch_index):
            self.stash_current(batch_index)

    def stash_current(self, batch_index):
        self.theta_buffer[batch_index] = self.create_current_cloned_buff()

    def _restore_from_buff(self, buff):
        with torch.no_grad():
            for pg, cloned in zip(self.optimizer.param_groups, buff):
                for p, bp in zip(pg['params'], cloned):
                    p.data = bp.data

    def pop_restore_stashed(self, batch_index):
        """ 
        Changed weight back to stashed wieghts.
        pops the stashed weights from memory.
        """
        _, buff = self.theta_buffer.pop(batch_index)
        self._restore_from_buff(buff)

    def restore_last(self):
        """
        Restore the the last inserted weights.
        """
        _, buff = self.theta_buffer.popitem(last=True)
        self._restore_from_buff(buff)

    def get_stashed_buff(self, batch_index, default=None):
        return self.theta_buffer.get(batch_index, default)
