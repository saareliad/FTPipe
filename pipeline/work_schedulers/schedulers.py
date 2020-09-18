import abc


class WorkScheduler(abc.ABC):
    def __init__(self, step_every, *args, **kw):
        self.step_every = step_every

    @abc.abstractmethod
    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds) -> bool:
        raise NotImplementedError()

    def reset(self):
        pass


class FBScheduler(WorkScheduler):
    """ Note: this is not like the scheduler in pipedream.
        In pipedream all partitions except last do D forwards in "warmup state",
        here every partitions does a different number of forwards in "warmup state" 
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds):
        assert 0 <= stage_depth < pipeline_depth

        # Last stage
        if stage_depth == 0:
            return True

        if done_fwds == num_batches:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1
        # TODO: we may want to allow more staleness when step_every > 1.

        return delta <= stage_depth


class VirtualStagesFBScheduler(FBScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.supremum_staleness = kw['supremum_staleness']
        self.num_gpus = kw['num_gpus']
        # self.stage_depth = kw['stage_depth']

    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds):
        """ Requires conversion to virtual stage to be done by caller e.g by get_virtual_stage_depth"""
        assert 0 <= stage_depth < pipeline_depth

        # Convert virtual: done by user
        # stage_virtual_depth = max(0, stage_depth - self.supremum_staleness + 1)

        # Last stage
        if stage_depth == 0:
            return True

        if done_fwds == num_batches:
            return False

        delta = done_fwds - done_bwds
        # TODO: we may want to allow more staleness when step_every > 1.

        return delta <= stage_depth

    def get_virtual_stage_depth(self, stage_depth: int) -> int:
        return max(0, stage_depth - self.supremum_staleness + 1)


class PipeDream1F1BScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.warmup = True

    def set_warmup(self, warmup=True):
        self.warmup = warmup

    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds):
        assert 0 <= stage_depth < pipeline_depth

        # Last stage
        if stage_depth == 0:
            return True

        if done_fwds == num_batches:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1
        if done_fwds == 0:
            self.warmup = True

        # Reached
        if delta == pipeline_depth:  # FIXME?
            self.warmup = False

        if self.warmup:
            return True

        return delta <= stage_depth

    def reset(self):
        self.warmup = True


class SeqScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds):

        if stage_depth == 0:
            return True

        if done_fwds == num_batches:
            return False

        return done_bwds == done_fwds


class Synchronous1F1BScheduler(WorkScheduler):
    """ "1f1b-gpipe.
        First scheduler I implemented in simulation 1.5 years ago...
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert hasattr(self, "step_every")

    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds):
        # NOTE: num_batches is number of batches
        num_micro_batches = self.step_every

        # Supports shorter "last batch"
        # User responsibility to check that
        # (1) last_batch_size % (normal_batch_size // step_every) == 0
        # (2) normal_batch_size % step_every == 0
        if done_fwds == num_batches:
            return False
        fwd_batch_idx = done_fwds // num_micro_batches
        bwd_batch_idx = done_bwds // num_micro_batches

        if fwd_batch_idx == bwd_batch_idx:
            # do 1F1B with micro batches, no staleness
            # Last stage
            return stage_depth == 0 or (done_fwds - done_bwds <= stage_depth)
        else:
            return False  # wait for backward (synchronous)


class GpipeScheduler(WorkScheduler):
    """
        GPipe scheduler with num_micro_batches = step_every.
        Supports shorter "last batch".

        NOTE:
            User responsibility to check that
            (1) last_batch_size % (normal_batch_size // step_every) == 0
            (2) normal_batch_size % step_every == 0
            This can easly be done with dataloader set to given micro_batch_size,
            that is (normal_batch_size // step_every).

        Example:
            For step_every = 4 we get FFFFBBBBFFFFBBBB...
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert hasattr(self, "step_every")

    def __call__(self, stage_depth, pipeline_depth, num_batches, done_fwds, done_bwds):
        # NOTE: num_batches is number of batches
        num_micro_batches = self.step_every

        # Supports shorter "last batch"
        # User responsibility to check that
        # (1) last_batch_size % (normal_batch_size // step_every) == 0
        # (2) normal_batch_size % step_every == 0
        if done_fwds == num_batches:
            return False
        fwd_batch_idx = done_fwds // num_micro_batches
        bwd_batch_idx = done_bwds // num_micro_batches

        return fwd_batch_idx == bwd_batch_idx
