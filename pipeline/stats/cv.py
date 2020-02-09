from typing import List, Dict, NamedTuple
from .interface import Stats
from types import SimpleNamespace
# import logging


def fit_res_to_dict(fit_res) -> Dict:
    if isinstance(fit_res, NamedTuple):
        fit_res = fit_res._asdict()
    elif isinstance(fit_res, SimpleNamespace):
        fit_res = fit_res.__dict__
    # elif isinstance(fit_res, dict)
    #     pass

    return fit_res


class FitResult(SimpleNamespace):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.record = []

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get_avg(self):
        return self.sum / self.count

# def record(self, va
    # def record(self, val):
    #     self.record.append(val)


class AccuracyMeter(AverageMeter):
    def __init__(self):
        super().__init__()

    def update(self, val, n=1):
        """ just to supoort adding num correct instead of accuracy """
        self.sum += val
        self.count += n


class CVStats(Stats):
    """ Class to handle statistics collection for CV Tasks """
    FIT_RESULTS_CLASS = FitResult

    def __init__(self, record_loss_per_batch=False):
        # Stats
        self.fit_res = self.FIT_RESULTS_CLASS(**self.fit_result_init_dict())
        assert not (self.fit_res is None)
        self.epoch_loss = AverageMeter()
        self.epoch_acc = AccuracyMeter()

        self.epoch_meters = [self.epoch_loss, self.epoch_acc]

        self.record_loss_per_batch = record_loss_per_batch
        self.training = True

    def fit_result_init_dict(self):
        return dict(num_epochs=0, train_loss=[], train_acc=[], test_loss=[], test_acc=[])

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def on_batch_end(self, loss, num_correct, batch_size):
        if self.record_loss_per_batch:
            if self.training:
                self.fit_res.train_loss.append(loss)
            else:
                self.fit_res.test_loss.append(loss)

        self.epoch_loss.update(loss, batch_size)
        self.epoch_acc.update(num_correct, batch_size)

    def on_epoch_end(self):
        if self.training:
            if not self.record_loss_per_batch:
                self.fit_res.train_loss.append(self.epoch_loss.get_avg())

            self.fit_res.train_acc.append(self.epoch_acc.get_avg() * 100)
            # FIXME: its only here, currently assuming test are same as train.
            self.fit_res.num_epochs += 1
        else:
            if not self.record_loss_per_batch:
                self.fit_res.test_loss.append(self.epoch_loss.get_avg())

            self.fit_res.test_acc.append(self.epoch_acc.get_avg() * 100)

        for meter in self.epoch_meters:
            meter.reset()

    def non_latst_partition_on_epoch_end(self):
        pass  # FIXME:
        # for meter in self.epoch_meters:
        #     meter.reset()

    def get_stats(self, *args):
        return fit_res_to_dict(self.fit_res)

    def get_epoch_info_str(self, is_train):
        if is_train:
            name = "train"
            loss = self.fit_res.train_loss[-1]
            acc = self.fit_res.train_acc[-1]
        else:
            name = "valid"
            loss = self.fit_res.test_loss[-1]
            acc = self.fit_res.test_acc[-1]

        return ' | {} loss {:5.2f} | {} acc {:4.2f}'.format(name, loss, name, acc)


class FitResultWithGradNorm(FitResult):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the accuracies are per epoch.
    """
    num_epochs: int
    grad_norm: List[float]
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class NormCVstats(CVStats):
    FIT_RESULTS_CLASS = FitResultWithGradNorm

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.epoch_grad_norm_meter = AverageMeter()
        # self.epoch_meters.append(self.epoch_grad_norm_meter)
        assert not (self.fit_res is None)

    def fit_result_init_dict(self):
        return dict(grad_norm=[], **super().fit_result_init_dict())

    def on_batch_end(self, loss, num_correct, batch_size, grad_norm=None):
        # Note: This is also called for test
        super().on_batch_end(loss, num_correct, batch_size)

        # TODO: not sure fi thats the best way
        if self.training and (not (grad_norm is None)):
            # if self.record_loss_per_batch:
            #     self.fit_res.append(grad_norm)

            self.epoch_grad_norm_meter.update(grad_norm)

    def non_last_partition_on_batch_end(self, grad_norm):
        # Called just for train
        if self.training:
            self.update_statistic_after_batch("grad_norm", grad_norm)

        # if self.training:
        #     if not (grad_norm is None):
        #         self.epoch_grad_norm_meter.update(grad_norm)
        #     else:
        #         logger = logging.getLogger("msnag")
        #         logger.warning(
        #             f"-W- grad norm is None for a non last partition. updating as 0")
        #         self.epoch_grad_norm_meter.update(0)

    def on_epoch_end(self):
        if self.training:
            self.fit_res.grad_norm.append(
                self.epoch_grad_norm_meter.get_avg())
        super().on_epoch_end()

    def non_latst_partition_on_epoch_end(self):
        assert(self.training)
        self.fit_res.grad_norm.append(
            self.epoch_grad_norm_meter.get_avg())

        self.epoch_grad_norm_meter.reset()
        # super().non_latst_partition_on_epoch_end()

    # Removed it, because its useless to see just for last partition...
    # def get_epoch_info_str(self, is_train):
    #     if is_train:
    #         my_addition = ' | grad_norm {:6.3f}'.format(
    #             self.fit_res.grad_norm[-1])
    #         return super().get_epoch_info_str(is_train) + my_addition
    #     return super().get_epoch_info_str(is_train)

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
    and the accuracies are per epoch.
    """
    num_epochs: int
    grad_norm: List[float]
    gap: List[float]
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class CVDistanceNorm(NormCVstats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simple.
    FIT_RESULTS_CLASS = FitResultWithGradNormAndDistance

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_statistic("gap", AverageMeter())
        # self.epoch_gap_meter = AverageMeter()
        assert not (self.fit_res is None)

    def fit_result_init_dict(self):
        return dict(gap=[], **super().fit_result_init_dict())

    def non_latst_partition_on_epoch_end(self):
        assert(self.training)
        self.fit_res.gap.append(
            self.epoch_gap_meter.get_avg())
        self.epoch_gap_meter.reset()
        super().non_latst_partition_on_epoch_end()

    # Removed it, because its useless to see just for last partition...
    # def get_epoch_info_str(self, is_train):
    #     if is_train:
    #         my_addition = ' | gap {:6.3f}'.format(
    #             self.fit_res.gap[-1])
    #         return super().get_epoch_info_str(is_train) + my_addition
    #     return super().get_epoch_info_str(is_train)

    def get_stats(self, stage_id):
        fit_res = super().get_stats(stage_id)
        new_name = f"p{stage_id}_gap"
        old_name = 'gap'
        fit_res[new_name] = fit_res.pop(old_name)
        return fit_res


class FitResultWithDistance(FitResult):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch (or per epoch, depends on config)
    and the accuracies are per epoch.
    """
    num_epochs: int
    gap: List[float]
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


# Code copy from ^
class CVDistance(CVStats):
    # FIXME: This whole chain of classes has HORRIBLE design. just implement it simple.
    FIT_RESULTS_CLASS = FitResultWithDistance

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.add_statistic("gap", AverageMeter())
        # self.epoch_gap_meter = AverageMeter()
        assert not (self.fit_res is None)

    def fit_result_init_dict(self):
        return dict(gap=[], **super().fit_result_init_dict())

    def non_latst_partition_on_epoch_end(self):
        assert(self.training)
        self.fit_res.gap.append(
            self.epoch_gap_meter.get_avg())
        self.epoch_gap_meter.reset()
        super().non_latst_partition_on_epoch_end()

    # Removed it, because its useless to see just for last partition...
    # def get_epoch_info_str(self, is_train):
    #     if is_train:
    #         my_addition = ' | gap {:6.3f}'.format(
    #             self.fit_res.gap[-1])
    #         return super().get_epoch_info_str(is_train) + my_addition
    #     return super().get_epoch_info_str(is_train)

    def get_stats(self, stage_id):
        fit_res = super().get_stats(stage_id)
        new_name = f"p{stage_id}_gap"
        old_name = 'gap'
        fit_res[new_name] = fit_res.pop(old_name)
        return fit_res
