from .interface import Stats
from .utils import AverageMeter


class SquadStats(Stats):
    """ Class to handle statistics collection for Squad Tasks """
    def __init__(self, record_loss_per_batch=False, is_last_partition=True):
        # Stats
        super().__init__(is_last_partition=is_last_partition)

        self.add_statistic(
            name="loss",
            meter=AverageMeter(),
            per_batch=record_loss_per_batch,
            per_epoch=not record_loss_per_batch,  # FIXME
            train=True,
            test=True)

        self.record_loss_per_batch = record_loss_per_batch

    def get_epoch_info_str(self, is_train):
        # FIXME: in per-batch-loss it returns value for the last batch instead of for epoch!
        if is_train:
            name = "train"
            loss = self.fit_res.train_loss[-1]
        else:
            name = "valid"
            loss = self.fit_res.test_loss[-1]
            # TODO: squad eval

        return ' | {} loss {:5.2f}'.format(
            name, loss, name)


class NormSquadstats(SquadStats):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.add_statistic(
            name="grad_norm",
            meter=AverageMeter(),
            per_batch=self.record_loss_per_batch,
            per_epoch=not self.record_loss_per_batch,  # FIXME
            train=True,
            test=False)
        self.register_pipeline_per_stage_statistic("grad_norm")


class SquadDistanceNorm(NormSquadstats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simple.

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_statistic(name="gap",
                           meter=AverageMeter(),
                           per_batch=self.record_loss_per_batch,
                           per_epoch=not self.record_loss_per_batch,
                           train=True,
                           test=False)
        self.register_pipeline_per_stage_statistic("gap")


class SquadDistance(SquadStats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simply.

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.add_statistic(
            name="gap",
            meter=AverageMeter(),
            per_batch=self.record_loss_per_batch,
            per_epoch=not self.record_loss_per_batch,  # FIXME
            train=True,
            test=False)
        self.register_pipeline_per_stage_statistic("gap")
