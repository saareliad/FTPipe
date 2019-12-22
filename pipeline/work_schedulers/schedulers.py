import abc


class WorkScheduler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        raise NotImplementedError()

# from .. import partition_manager as pmgr  # import SinglePartitionManager
# class MgrAwareWorkScheduler(WorkScheduler):
#     def __init__(self, mgr: pmgr.SinglePartitionManager):
#         self.mgr = mgr


class FBScheduler(WorkScheduler):

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        # Last stage
        if stage == num_stages - 1:
            return True

        if done_fwds == num_steps:
            return False

        delta = done_fwds - done_bwds
        # allowed_staleness = num_stages-stage-1

        return delta < num_stages-stage


class SeqScheduler(WorkScheduler):
    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        if stage == num_stages - 1:
            return True

        if done_fwds == num_steps:
            return False

        return done_bwds == done_fwds


class GpipeScheduler(WorkScheduler):
    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        if done_fwds == num_steps:
            return False

        f = done_fwds // num_stages
        b = done_bwds // num_stages

        return f == b


if __name__ == "__main__":
    num_stages = 4
    # stage = 0  # Should test the edge case.
    num_batches = 9

    def print_for_stage(stage):
        f = 0
        b = 0
        scheduler = FBScheduler()
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
        print(s)

    for stage in range(num_stages):
        print(f"Stage {stage}")
        print_for_stage(stage)
        print()

