import types
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

    def __init__(self, optimizer, step_every=1, has_weight_predictor=False, true_weights_storage=None):
        self.optimizer = optimizer
        self.theta_buffer = OrderedDict()
        self.dirty_mark = OrderedDict()
        self.micro_batch = OrderedDict()
        self.temporery_short_term_buff = []
        # step every parameter, used to infer micro batch.
        self.step_every = step_every
        self.is_problematic = False
        self.has_weight_predictor = has_weight_predictor
        # TODO: reduce redundent stashing for micro batches.

        self.true_weights_storage = true_weights_storage

    def set_problematic(self, forward=True, policy='EVERY_BATCH'):
        self.is_problematic = True
        # POLICIES = {'CHANGE', 'EVERY_BATCH'}
        se = self.step_every
        if policy == 'CHANGE':
            def get_micro_batch(self, batch_index):
                # TODO: can do a more "fine grained condition"
                # saves computation for earlier partitions:
                # L - num partitions
                # roundtrip = 2*L - 1
                # min(initial_forwards, roundtrip) + (step.every - 1)
                if batch_index <= se:
                    return batch_index
                return (batch_index+1) % se
        elif policy == 'EVERY_BATCH':
            def get_micro_batch(self, batch_index):
                # return 0  # TODO: this can be improved, but I have problems with the dirty...
                return batch_index if batch_index < se else 0
        else:
            raise NotImplementedError()

        if forward:
            self.get_micro_batch_forward = types.MethodType(
                get_micro_batch, self)
        else:
            self.get_micro_batch_backward = types.MethodType(
                get_micro_batch, self)

    def get_micro_batch_forward(self, batch_index):
        return batch_index % self.step_every

    def get_micro_batch_backward(self, batch_index):
        return batch_index % self.step_every

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
        """
        Stashes current weights if we expect updates.
        Also tracks "dirty-ness" of real weights w.r.t given batch_index
        # TODO: option to set dirty right ahead!
        """
        # print(f"stashed {batch_index}, rank {rank}")
        # if rank == 0:
        #     print(
        #         f"Stash? fwd_batch_idx:{batch_index}, expected_updates:{expected_updates}, \
        # mb: {batch_index % self.step_every}")
        #     print(self.dirty_mark)
        # Aviod stashing in case of no staleness.
        if expected_updates > 0:

            # HACK: we use the same buffer for differnt micro batches!
            # we can do it because we don't step on stashed wieghts.
            micro_batch = self.get_micro_batch_forward(batch_index)
            buff = self.theta_buffer.get(
                batch_index - 1, None) if micro_batch > 0 else None

            if buff is None:
                buff = self._create_current_cloned_buff()
            else:
                # assert not (self.dirty_mark[batch_index-1])  # check for bug...
                if (self.dirty_mark[batch_index-1]) and not self.has_weight_predictor:
                    s = f"Attemted to use dirty buff as stash: rank:\
                        {rank} b:{batch_index}, mb:{micro_batch}, prev b:{batch_index-1} \
                          expected_updates:{expected_updates}"
                    raise RuntimeError(s)

            self.theta_buffer[batch_index] = buff

            if self.has_weight_predictor:
                # So we post restore true weights!
                self.dirty_mark[batch_index] = True
            else:
                self.dirty_mark[batch_index] = False
        else:
            # HACK: mark as not dirty anyway.
            self.dirty_mark[batch_index] = False

    def _restore_from_buff(self, buff):
        # "Copy" pointers.
        # NOTE: its ok to do so, because we only stash once.
        with torch.no_grad():
            for pg, cloned in zip(self.optimizer.param_groups, buff):
                for p, bp in zip(pg['params'], cloned):
                    p.data = bp.data

    # def ensure_correct_post_restore(self, batch_idx):
    #     """ Used before `pop_restore_stashed()` """
    #     # This functionality can be simply described as:
    #     #   if did step since I was stasshed =>
    #     #       create temporary buffer -> (do backward) -> ... -> (do step)-> then restore it
    #     # at batch_index=0, Add the temp buff for the one which will step

    #     # the in case micro batches is used, the temp_buffer is duplicated at mb=0 for all of them.

    #     # Extra Details:
    #     # problem this function is supposed to solve:
    #     # what we are about to do is:
    #     # restore stashed wieghts.data
    #     # compute backward on them
    #     # restore back to current, up to date weights.data, with restore_last().
    #     # however, there is possibility that current `batch_index` is not the last.
    #     # (This can happen in case of several backwards one after another)
    #     # In this case, we need to stash the currect version of weight so it will be the last,
    #     # and this is what the function does.
    #     # Note we use temp buffer because we are about to pop from the original buffer
    #     if self.dirty_mark[batch_idx] and (batch_idx % self.step_every == 0):
    #         self.temporery_short_term_buff.append(self._create_current_cloned_buff())

    def pop_restore_stashed(self, batch_index):
        """ 
        Changed weight back to stashed wieghts.
        pops the stashed weights from memory.

        (used before backward, 
        and after `ensure_correct_post_restore(...)` was called)
        """
        # print(f"popped {batch_index} rank {rank}")
        is_dirty = self.dirty_mark.pop(batch_index)
        if is_dirty and (self.get_micro_batch_backward(batch_index) == 0):
            # Ensure correct post resotore:
            # create temporary buffer -> (do backward) -> ... -> (do step)-> then restore it
            # Same versions as backward!
            # NOTE: where ws+msnag is used, this is "copy on write"
            self.true_weights_storage.create_cloned_if_needed()
        if is_dirty:
            buff = self.theta_buffer.pop(batch_index)
            self._restore_from_buff(buff)
            # tell self.true_weights_storage thats we are in stashed more.
            self.true_weights_storage.record_change_mode("stashed")
        else:
            assert batch_index not in self.theta_buffer

    # Exposed for statistics and alike
    def get_stashed_buff(self, batch_index, default=None):
        return self.theta_buffer.get(batch_index, default)

    def pop_stashed_buff(self, batch_index):
        return self.theta_buffer.pop(batch_index, None)