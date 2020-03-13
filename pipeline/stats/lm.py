from typing import List
from .interface import Stats
from types import SimpleNamespace
import math
from .utils import fit_res_to_dict, AverageMeter


class FitResult(SimpleNamespace):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the PPLS are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_ppl: List[float]
    test_loss: List[float]
    test_ppl: List[float]


class LMStats(Stats):
    """ Class to handle statistics collection for LM Tasks """
    FIT_RESULTS_CLASS = FitResult

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

    def fit_result_init_dict(self):
        return dict(num_epochs=0,
                    train_loss=[],
                    train_ppl=[],
                    test_loss=[],
                    test_ppl=[])

    def last_partition_on_batch_end(self, loss, batch_size):
        # TODO: maby pass this dict so we can use inheritance
        d = {"loss": (loss, batch_size), "ppl": (math.exp(loss), batch_size)}

        self.update_fit_res_after_batch_all(d)
        self.update_statistic_after_batch_all(d)

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


class FitResultWithGradNorm(FitResult):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the ppls are per epoch.
    """
    num_epochs: int
    grad_norm: List[float]
    train_loss: List[float]
    train_ppl: List[float]
    test_loss: List[float]
    test_ppl: List[float]


class NormLMstats(LMStats):
    FIT_RESULTS_CLASS = FitResultWithGradNorm

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.add_statistic(
            name="grad_norm",
            meter=AverageMeter(),
            per_batch=self.record_loss_per_batch,
            per_epoch=not self.record_loss_per_batch,  # FIXME
            train=True,
            test=False)

    def fit_result_init_dict(self):
        return dict(grad_norm=[], **super().fit_result_init_dict())

    def last_partition_on_batch_end(self, loss, batch_size, grad_norm=None):
        # Note: This is also called for test
        super().last_partition_on_batch_end(loss, batch_size)

        d = {"grad_norm": (grad_norm, 1)}

        self.update_fit_res_after_batch_all(d)
        self.update_statistic_after_batch_all(d)

    def non_last_partition_on_batch_end(self, grad_norm):
        # if self.training:
        super().non_last_partition_on_batch_end()
        d = {"grad_norm": (grad_norm, 1)}
        self.update_fit_res_after_batch_all(d)
        self.update_statistic_after_batch_all(d)

    def get_stats(self, stage_id=None):
        fit_res = super().get_stats()
        if not (stage_id is None):
            new_name = f"p{stage_id}_grad_norm"
            old_name = 'grad_norm'
            fit_res[new_name] = fit_res.pop(old_name)
        return fit_res


class FitResultWithGradNormAndDistance(FitResult):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the PPLs are per epoch.
    """
    num_epochs: int
    grad_norm: List[float]
    gap: List[float]
    train_loss: List[float]
    train_ppl: List[float]
    test_loss: List[float]
    test_ppl: List[float]


class LMDistanceNorm(NormLMstats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simple.
    FIT_RESULTS_CLASS = FitResultWithGradNormAndDistance

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_statistic(name="gap",
                           meter=AverageMeter(),
                           per_batch=self.record_loss_per_batch,
                           per_epoch=not self.record_loss_per_batch,
                           train=True,
                           test=False)

    def fit_result_init_dict(self):
        return dict(gap=[], **super().fit_result_init_dict())

    def get_stats(self, stage_id):
        fit_res = super().get_stats(stage_id)
        new_name = f"p{stage_id}_gap"
        old_name = 'gap'
        fit_res[new_name] = fit_res.pop(old_name)
        return fit_res

    # TODO:
    # def non_last_partition_on_batch_end(self,):
    # def last_partition_on_batch_end(self,):


class FitResultWithDistance(FitResult):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the PPLS are per epoch.
    """
    num_epochs: int
    gap: List[float]
    train_loss: List[float]
    train_ppl: List[float]
    test_loss: List[float]
    test_ppl: List[float]


# Code copy from ^
class LMDistance(LMStats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simply.
    FIT_RESULTS_CLASS = FitResultWithDistance

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.add_statistic(
            name="gap",
            meter=AverageMeter(),
            per_batch=self.record_loss_per_batch,
            per_epoch=not self.record_loss_per_batch,  # FIXME
            train=True,
            test=False)

    def fit_result_init_dict(self):
        return dict(gap=[], **super().fit_result_init_dict())

    def get_stats(self, stage_id):
        fit_res = super().get_stats(stage_id)
        new_name = f"p{stage_id}_gap"
        old_name = 'gap'
        fit_res[new_name] = fit_res.pop(old_name)
        return fit_res
