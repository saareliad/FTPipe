from .interface import Stats
from .utils import AverageMeter
import math

class PPLMeter(AverageMeter):
    """ Update like loss, get_avg() gets the PPL """
    def get_avg(self):
        # avg_loss = super().get_avg()
        avg_loss = self.sum / self.count
        # ppl = math.exp(avg_loss)
        return math.exp(avg_loss)

class LMStats(Stats):
    """ Class to handle statistics collection for LM Tasks """
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

        self.add_statistic(
            name="ppl",
            meter=PPLMeter(),
            per_batch=False,
            per_epoch=True,  # FIXME
            train=True,
            test=True)

        self.record_loss_per_batch = record_loss_per_batch

    def get_epoch_info_str(self, is_train):
        # FIXME: in per-batch-loss it returns value for the last batch instead of for epoch!
        if is_train:
            name = "train"
            loss = self.fit_res.train_loss[-1]
            ppl = self.fit_res.train_ppl[-1]
        else:
            name = "valid"
            loss = self.fit_res.test_loss[-1]
            ppl = self.fit_res.test_ppl[-1]

        return ' | {} loss {:5.2f} | {} ppl {:4.3f}'.format(
            name, loss, name, ppl)


class NormLMstats(LMStats):
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


class LMDistanceNorm(NormLMstats):
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


class LMDistance(LMStats):
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
