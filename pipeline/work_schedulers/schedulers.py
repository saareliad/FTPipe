import abc


class WorkScheduler(abc.ABC):
    def __init__(self, step_every):
        self.step_every = step_every

    @abc.abstractmethod
    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        raise NotImplementedError()

    def reset(self):
        pass

# from .. import partition_manager as pmgr  # import SinglePartitionManager
# class MgrAwareWorkScheduler(WorkScheduler):
#     def __init__(self, mgr: pmgr.SinglePartitionManager):
#         self.mgr = mgr


class FBScheduler(WorkScheduler):
    """ Note: this is not like the scheduler in pipedream.
        In pipedream all partititions excpet last do D forwards in "warmup state",
        here every partitions does a different number of forwards in "warmup state" 
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        # Last stage
        if stage == num_stages - 1:
            return True

        if done_fwds == num_steps:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1
        # TODO: we want to allow more stalenss when step_every > 1.

        return delta < num_stages-stage


class PipeDream1F1BScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.warmup = True

    def set_warmup(self, warmup=True):
        self.warmup = warmup

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        # Last stage
        if stage == num_stages - 1:
            return True

        if done_fwds == num_steps:
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

        return delta < num_stages-stage

    def reset(self):
        self.warmup = True


class SeqScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        if stage == num_stages - 1:
            return True

        if done_fwds == num_steps:
            return False

        return done_bwds == done_fwds


class GpipeScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        if done_fwds == num_steps:
            return False

        f = done_fwds // num_stages
        b = done_bwds // num_stages

        return f == b


def get_fwds_between_first_step_from_str(s, step_every):
    import re
    from collections import Counter
    all_B_idexes = [m.start() for m in re.finditer('B', s)]
    first = all_B_idexes[step_every-1]
    second = all_B_idexes[2+(step_every-1)]
    c1 = Counter(s[:first])['F']
    c2 = Counter(s[:second])['F']
    idexes = list(range(c1, c2))
    return idexes


def print_for_stage(stage, scheduler, num_stages, num_batches):
    f = 0
    b = 0
    # scheduler = PipeDream1F1BScheduler()
    # scheduler = FBScheduler(step_every)
    s = ""
    while b < num_batches:
        if scheduler(stage, num_stages, num_batches, f, b):
            s += "F"
            f += 1
            if stage == num_stages - 1:
                s += "B"
                b += 1
        else:
            s += "B"
            b += 1
    # print(s)
    scheduler.reset()
    return s


def get_fwds_between_first_and_seconds_step_for_stage(scheduler, stage, num_stages, num_batches):
    s = print_for_stage(stage, scheduler, num_stages, num_batches)
    step_every = scheduler.step_every
    fwds = get_fwds_between_first_step_from_str(s, step_every)
    is_problematic = fwds[0] % step_every !=0
    return fwds, is_problematic



if __name__ == "__main__":
    num_stages = 4
    EXTRA = 5
    # stage = 0  # Should test the edge case.
    num_batches = num_stages*2 + 1 + EXTRA
    step_every = 2

    sched_name = "1F1B"
    AVAILABLE_WORK_SCHEDULERS = {"1F1B": FBScheduler,
                                 "SEQ": SeqScheduler,
                                 "GPIPE": GpipeScheduler,
                                 "PIPEDREAM": PipeDream1F1BScheduler}

    scheduler = AVAILABLE_WORK_SCHEDULERS.get(sched_name)(step_every)

    def print_for_all_stages(num_stages):
        stage_strings = dict()  # just for pretty printing
        for stage in range(num_stages):
            print(f"Stage {stage}")
            s = print_for_stage(stage, scheduler, num_stages, num_batches)
            print(s)
            stage_strings[stage] = s
            print()

    print_for_all_stages(num_stages)

    stage_fwds_between_first_step = dict()
    stage_fwds_problematic = []
    for stage in range(num_stages):
        fwds, is_problematic = get_fwds_between_first_and_seconds_step_for_stage(
            scheduler, stage, num_stages, num_batches)
        stage_fwds_between_first_step[stage] = fwds
        if is_problematic:
            stage_fwds_problematic.append(stage)

    print()
    print("Stage_fwds_between_first_step:")
    print(stage_fwds_between_first_step)
    print()
    print("Problematic stages:", stage_fwds_problematic)
