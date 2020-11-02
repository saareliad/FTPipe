import os
import types
from collections import OrderedDict
from enum import Enum, auto

import torch

rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))


class CachePolicy(Enum):
    EVERY_BATCH = auto()
    STEP_EVERY = auto()


# TODO: can analyze when doing gap aware is possible

class WeightStasher:
    """ Helper calss to handle weight stashing
    API:
        Stash during FWD pass:
            stash_current(idx)

        Pre backward pass:
            pop_and_load_stashed_params(idx)

        Post backward pass:
            # back to true weights

    # TODO: look to pipedream implementation and understand if they did something special with the weight decay.
    # TODO: think about batch norm, and similar "dangerous" layers.

    """

    def __init__(self,
                 optimizer,
                 step_every=1,
                 has_weight_predictor=False,
                 true_weights_storage=None,
                 using_clone_weight_predictor=False):
        self.optimizer = optimizer
        self.theta_buffer = OrderedDict()
        self.dirty_mark = OrderedDict()
        self.micro_batch = OrderedDict()
        self.temporery_short_term_buff = []
        # step every parameter, used to infer micro batch.
        self.step_every = step_every
        self.is_problematic = False
        self.has_weight_predictor = has_weight_predictor
        self.using_clone_weight_predictor = using_clone_weight_predictor

        self.true_weights_storage = true_weights_storage
        # TODO: reduce redundant stashing for micro batches. (check do simply at init...)

    def set_problematic(self, stage_depth, pipeline_depth, forward=True, policy: CachePolicy = CachePolicy.EVERY_BATCH):

        self.is_problematic = True
        se = self.step_every
        if policy == CachePolicy.STEP_EVERY:
            if se >= stage_depth:
                # FIXME: whole usage of num_stages
                def get_micro_batch(self, batch_index):
                    true_mb = batch_index % se
                    if true_mb <= stage_depth + (se - pipeline_depth):  # FIXME
                        return true_mb  # staleness 0
                    else:
                        return true_mb - (stage_depth + (se - pipeline_depth) + 1)  # staleness 1  # FIXME

                    # TODO: can do a more "fine grained condition"
                    # saves computation for earlier partitions:
                    # L - num partitions
                    # roundtrip = 2*L - 1
                    # min(initial_forwards, roundtrip) + (step.every - 1)
                    # if batch_index <= se:
                    #     return batch_index
                    # return (batch_index + 1) % se
            else:
                # TODO: not so sure
                return  # no change
                # def get_micro_batch(self, batch_index):
                #     return batch_index % se
            # else:
            #     warnings.warn(f"Did not implement better caching so setting EVERY_BATCH caching weight stashing policy instead of STEP_EVERY for stage_id {stage_id}")
            #     return self.set_problematic(stage_id,num_stages,forward=forward,policy=CachePolicy.EVERY_BATCH)
            #     # raise NotImplementedError()

        elif policy == CachePolicy.EVERY_BATCH:
            def get_micro_batch(self, batch_index):
                # return 0  # TODO: this can be improved, but I have problems with the dirty...
                return batch_index if batch_index < se else 0
        else:
            raise NotImplementedError()

        if forward:
            self.get_micro_batch_forward = types.MethodType(
                get_micro_batch, self)
        else:
            raise NotImplementedError()

    def get_micro_batch_forward(self, batch_index):
        return batch_index % self.step_every

    def mark_stashed_as_dirty(self):
        for i in self.dirty_mark:
            self.dirty_mark[i] = True

    # TODO: this may cause the problem with aggregation.
    def _create_current_cloned_buff(self):
        # TODO: we can completely avoid the clone for clone weight prediction.
        # becasue, we
        # (1) first clone() for weight prediction,
        # (2) then stash the cloned weight.
        if self.using_clone_weight_predictor:
            with torch.no_grad():
                buff = [[p for p in pg['params']]
                        for pg in self.optimizer.param_groups]
        else:
            with torch.no_grad():
                buff = [[p.detach().clone() for p in pg['params']]
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
        if expected_updates > 0:

            # HACK: we use the same buffer for different micro batches!
            # we can do it because we don't step on stashed weights.
            micro_batch = self.get_micro_batch_forward(batch_index)
            buff = self.theta_buffer.get(batch_index -
                                         1, None) if micro_batch > 0 else None

            if buff is None:
                buff = self._create_current_cloned_buff()
            else:
                # assert not (self.dirty_mark[batch_index-1])  # check for bug...
                if (self.dirty_mark[batch_index -
                                    1]) and not self.has_weight_predictor:
                    s = f"Attemted to use dirty buff as stash: rank:\
                        {rank} b:{batch_index}, mb:{micro_batch}, prev b:{batch_index - 1} \
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
                pgp = pg['params']
                for p, bp in zip(pgp, cloned):
                    p.data = bp.detach()

    def pop_and_load_stashed_params(self, batch_index):
        """
        Changed weight back to stashed weights.
        pops the stashed weights from memory.

        (used before backward)
        """
        is_dirty = self.dirty_mark.pop(batch_index)
        if is_dirty:
            # create temporary buffer for true weights
            # load stashed ->
            # (do backward) ->
            # restore true weights ->
            # (do step)
            self.true_weights_storage.create_cloned_if_needed()
            buff = self.theta_buffer.pop(batch_index)
            self._restore_from_buff(buff)
            self.true_weights_storage.record_change_mode("stashed")
        else:
            assert batch_index not in self.theta_buffer

    # Exposed for statistics and alike
    def get_stashed_buff(self, batch_index, default=None):
        return self.theta_buffer.get(batch_index, default)

    def pop_stashed_buff(self, batch_index):
        return self.theta_buffer.pop(batch_index, None)
