import abc
import pprint
import pandas as pd


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


class StepEvery1F1BScheduler(WorkScheduler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        raise NotImplementedError()


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
    second = all_B_idexes[2*step_every-1]
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

# def get_batch_to_micro_batch_dicts(stage, scheduler, num_stages, num_batches, use_wp=False):
#     step_every =  scheduler.step_every
#     d = get_staleness_for_stage(
#                     stage, scheduler, num_stages, num_batches, step_every)
#     # to list of records...

#     df = pd.DataFrame.from_record(list(d.values()))
#     l = df.groupby('mv').apply(lambda df: len(df)).to_dict()
#     df['unique_mv'] = list(map(i.get, df['mv']))
#     df['unique_mv_cs'] = df['unique_mv'].cumsum()


# 0,1,2,3 -> 2


def get_fwds_between_first_and_seconds_step_for_stage(scheduler, stage, num_stages, num_batches):
    s = print_for_stage(stage, scheduler, num_stages, num_batches)
    step_every = scheduler.step_every
    fwds = get_fwds_between_first_step_from_str(s, step_every)
    # These fwds have the same "mv" (forward version)
    # now, we need to check if thier "bv" (backward versions)

    if len(fwds) == 1:
        is_problematic = False
    else:
        # bv = {backward_version(i, 1, step_every)
        #       for i in range(fwds[0], fwds[-1]+1)}

        # or expected versions "ev" are the same.
        # FIXME:
        # d = get_staleness_for_stage(stage, scheduler, num_stages, num_batches, step_every)
        is_problematic = fwds[0] % step_every != 0  # fwds[0] % step_every != 0

    return fwds, is_problematic

# def get_micro_batch(batch_idx, se):
#     return batch_idx % se


def should_do_step(batch_idx, se):
    do_step = (batch_idx % se) == (se-1)
    return do_step


def expected_staleness(done_fwds, done_bwds, se):
    return sum([should_do_step(x, se) for x in range(done_bwds, done_fwds)])


def my_version(done_bwds, se):
    """ steps so far """
    return sum([should_do_step(i, se) for i in range(done_bwds)])


def expected_version(done_fwds, done_bwds, se):
    # Tuple: current + weight prediction
    return (my_version(done_bwds, se), expected_staleness(done_fwds, done_bwds, se))


def backward_version(done_fwds, done_bwds, se):
    return my_version(done_bwds, se) + expected_staleness(done_fwds, done_bwds, se)
    # stage 0, len 4: [expected_staleness(i, max(i-3,0), 2) for i in range(10)]


def get_staleness_for_stage(stage, scheduler, num_stages, num_batches, se):
    s = print_for_stage(stage, scheduler, num_stages, num_batches)
    d = {}
    done_fwds = 0
    done_bwds = 0
    for c in s:
        if c == 'F':
            es = expected_staleness(done_fwds, done_bwds, se)
            mv = my_version(done_bwds, se)
            ev = expected_version(done_fwds, done_bwds, se)
            bv = mv + es
            mys = s[:done_fwds+done_bwds+1]
            d[done_fwds] = dict(es=es, mv=mv, ev=ev, bv=bv, mys=mys)
            done_fwds += 1
        if c == 'B':
            steps_so_far = done_bwds // se
            if not (steps_so_far == d[done_bwds]['bv']):
                # pprint.pprint(d)
                raise AssertionError(f"Stage:{stage}, batch:{done_bwds}, steps_so_far:{steps_so_far},\
                    but predicted: {d[done_bwds]['bv']}.\n \
                    Extra:\n {pprint.pformat(d)},\n {s[:done_fwds+done_bwds+1]}")

            done_bwds += 1

    return d


# print([expected_version(i, max(i-3,0), 2) for i in range(15)]); print(list(range(15))); print([expected_staleness(i, max(i-3,0), 2) for i in range(15)]);


if __name__ == "__main__":
    # TODO: automatic tests
    num_stages = 4
    EXTRA = 20
    # stage = 0  # Should test the edge case.
    num_batches = num_stages*2 + 1 + EXTRA
    step_every = 3
    sched_name = "1F1B"
    PRINT_STAGE_STRINGS = False
    PRINT_EXTRA_INFO = True
    PRINT_JUST_PROBLEMATIC_STAGES = True
    PRINT_JUST_FOR_THE_FWDS = False

    d = dict(sched_name=sched_name,
             step_every=step_every, num_stages=num_stages)
    print(f"-I- got args: {pprint.pformat(d)}")
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

    if PRINT_STAGE_STRINGS:
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

    if PRINT_EXTRA_INFO:
        for stage in range(num_stages):
            print(f"Stage {stage}")
            d = get_staleness_for_stage(
                stage, scheduler, num_stages, num_batches, step_every)
            if PRINT_JUST_PROBLEMATIC_STAGES:
                if stage not in stage_fwds_problematic:
                    continue
                if PRINT_JUST_FOR_THE_FWDS:
                    d = {i: d[i] for i in stage_fwds_between_first_step[stage]}
            pprint.pprint(d)
