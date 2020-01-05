import torch
from collections import OrderedDict

import os
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))


class WeightStasher:
    """ Helper calss to handle weight stashing
    API:
        Stash during FWD pass:
            stash_current(idx)

        Pre backward pass:
            ensure_correct_post_restore(idx)
            pop_restore_stashed(idx)

        Post backward pass:
            post_restore()  # back to normal weights

    # TODO: look to pipedream implementation and udnerstand if they did something special with the weight decay.
    # TODO: think about batch norm, and simillar "dangerous" layers.

    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.theta_buffer = OrderedDict()
        self.dirty_mark = OrderedDict()
        self.temporery_short_term_buff = []

    def mark_stashed_as_dirty(self):
        for i in self.dirty_mark:
            self.dirty_mark[i] = True

    def _create_current_cloned_buff(self):
        with torch.no_grad():
            buff = [[p.data.clone() for p in pg['params']]
                    for pg in self.optimizer.param_groups]
        return buff

    def _is_current_last(self, batch_index):
        return next(reversed(self.theta_buffer.keys())) == batch_index

    def stash_current(self, batch_index, expected_updates):
        # print(f"stashed {batch_index}, rank {rank}")
        # Aviod stashing in case of no staleness.
        if expected_updates > 0:
            self.theta_buffer[batch_index] = self._create_current_cloned_buff()

        self.dirty_mark[batch_index] = False  # HACK: mark as not dirty anyway.

    def _restore_from_buff(self, buff):
        with torch.no_grad():
            for pg, cloned in zip(self.optimizer.param_groups, buff):
                for p, bp in zip(pg['params'], cloned):
                    p.data = bp.data

    def ensure_correct_post_restore(self, batch_idx):
        # This functionality can be simply described as:
        #   if did step since last call =>
        #       create temporary buffer and restore from it later.

        # Extra Details:
        # problem this function is supposed to solve:
        # what we are about to do is:
        # restore stashed wieghts.data
        # compute backward on them
        # restore back to current, up to date weights.data, with restore_last().
        # however, there is possibility that current `batch_index` is not the last.
        # (This can happen in case of several backwards one after another)
        # In this case, we need to stash the currect version of weight so it will be the last,
        # and this is what the function does.
        # Note we use temp buffer because we are about to pop from the original buffer

        if self.dirty_mark[batch_idx]:
            # print(f"Ensured for {batch_idx} rank {rank}")
            self.temporery_short_term_buff.append(
                self._create_current_cloned_buff())

    def pop_restore_stashed(self, batch_index):
        """ 
        Changed weight back to stashed wieghts.
        pops the stashed weights from memory.
        """
        # print(f"popped {batch_index} rank {rank}")
        dirty = self.dirty_mark.pop(batch_index)
        if dirty:
            buff = self.theta_buffer.pop(batch_index)
            self._restore_from_buff(buff)
        else:
            assert batch_index not in self.theta_buffer

    def post_restore(self, batch_index):
        if self.temporery_short_term_buff:
            buff = self.temporery_short_term_buff.pop()
            self._restore_from_buff(buff)

    # Exposed for statistics and alike
    def get_stashed_buff(self, batch_index, default=None):
        return self.theta_buffer.get(batch_index, default)
