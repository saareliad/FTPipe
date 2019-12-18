import abc


class WorkScheduler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        raise NotImplementedError()


class FBScheduler(WorkScheduler):

    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        if stage == num_stages:
            return True

        if done_fwds == num_steps:
            return False

        delta = done_fwds - done_bwds
        return delta <= num_stages-stage-1


class SeqScheduler(WorkScheduler):
    def __call__(self, stage, num_stages, num_steps, done_fwds, done_bwds):
        assert 0 <= stage < num_stages

        if stage == num_stages:
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
    stage = 1
    num_batches = 9
    f = 0
    b = 0
    scheduler = FBScheduler()
    s = ""
    while b < num_batches:
        if scheduler(stage, num_stages, num_batches, f, b):
            s += "F"
            f += 1
        else:
            s += "B"
            b += 1
    print(s)
