from .interface import Stats
import math
from .utils import fit_res_to_dict, AverageMeter


class LMStats(Stats):
    """ Class to handle statistics collection for LM Tasks """
    def __init__(self, record_loss_per_batch=False):
        # Stats
        super().__init__()

        self.add_statistic(
            name="loss",
            meter=AverageMeter(),
            per_batch=record_loss_per_batch,
            per_epoch=not record_loss_per_batch,  # FIXME
            train=True,
            test=True)

        self.add_statistic(
            name="ppl",
            meter=AverageMeter(),
            per_batch=False,
            per_epoch=True,  # FIXME
            train=True,
            test=True)

        self.record_loss_per_batch = record_loss_per_batch

    def get_stats(self, *args):
        return fit_res_to_dict(self.fit_res)

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

        return ' | {} loss {:5.2f} | {} ppl {:4.2f}'.format(
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

    def get_stats(self, stage_id=None):
        fit_res = super().get_stats()
        if not (stage_id is None):
            new_name = f"p{stage_id}_grad_norm"
            old_name = 'grad_norm'
            fit_res[new_name] = fit_res.pop(old_name)
        return fit_res


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

    def get_stats(self, stage_id):
        fit_res = super().get_stats(stage_id)
        new_name = f"p{stage_id}_gap"
        old_name = 'gap'
        fit_res[new_name] = fit_res.pop(old_name)
        return fit_res


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

    def get_stats(self, stage_id):
        fit_res = super().get_stats(stage_id)
        new_name = f"p{stage_id}_gap"
        old_name = 'gap'
        fit_res[new_name] = fit_res.pop(old_name)
        return fit_res
