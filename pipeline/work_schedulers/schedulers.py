import abc


class WorkScheduler(abc.ABC):
    def __init__(self, step_every, *args, **kw):
        self.step_every = step_every

    @abc.abstractmethod
    def __call__(self, stage, num_stages, num_batches, done_fwds,
                 done_bwds) -> bool:
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

    def __call__(self, stage, num_stages, num_batches, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        # Last stage
        if stage == num_stages - 1:
            return True

        if done_fwds == num_batches:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1
        # TODO: we may want to allow more staleness when step_every > 1.

        return delta < num_stages - stage


class VirtualStagesFBScheduler(FBScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.supremum_staleness = kw['supremum_staleness']
        self.num_gpus = kw['num_gpus']

    def __call__(self, stage, num_stages, num_batches, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        # Convert virtual:
        stage = max(0, stage - self.supremum_staleness)
        num_stages = num_stages - self.supremum_staleness

        # Last stage
        if stage == num_stages - 1:
            return True

        if done_fwds == num_batches:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1
        # TODO: we may want to allow more staleness when step_every > 1.

        return delta < min(self.supremum_staleness, num_stages - stage)


class PipeDream1F1BScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.warmup = True

    def set_warmup(self, warmup=True):
        self.warmup = warmup

    def __call__(self, stage, num_stages, num_batches, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        # Last stage
        if stage == num_stages - 1:
            return True

        if done_fwds == num_batches:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1
        if done_fwds == 0:
            self.warmup = True

        # Reached
        if delta == num_stages:
            self.warmup = False

        if self.warmup:
            return True

        return delta < num_stages - stage

    def reset(self):
        self.warmup = True


class SeqScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage, num_stages, num_batches, done_fwds, done_bwds):

        if stage == num_stages - 1:
            return True

        if done_fwds == num_batches:
            return False

        return done_bwds == done_fwds


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

    def __call__(self, stage, num_stages, num_batches, done_fwds, done_bwds):
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


class Synchronous1F1BScheduler(WorkScheduler):
    """ "1f1b-gpipe.
        First scheduler I implemented in simulation 1.5 years ago...
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert hasattr(self, "step_every")

    def __call__(self, stage, num_stages, num_batches, done_fwds, done_bwds):
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
            if stage == num_stages - 1:
                return True
            return stage == num_stages - 1 or (done_fwds - done_bwds < num_stages - stage)
        else:
            return False  # wait for backward (synchronous)
