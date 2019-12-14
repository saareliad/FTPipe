from typing import List
from .interface import Stats
from types import SimpleNamespace


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

    def __init__(self, record_loss_per_batch=False):
        # Stats
        self.fit_res = FitResult(
            num_epochs=0, train_loss=[], train_acc=[], test_loss=[], test_acc=[])
        self.epoch_loss = AverageMeter()
        self.epoch_acc = AccuracyMeter()

        self.fit_loss = AverageMeter()
        # self.fit_acc = AverageMeter()

        self.record_loss_per_batch = record_loss_per_batch
        self.training = True

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
            self.fit_res.num_epochs += 1  # FIXME: its only here, currently assuming test are same as train.
        else:
            if not self.record_loss_per_batch:
                self.fit_res.test_loss.append(self.epoch_loss.get_avg())

            self.fit_res.test_acc.append(self.epoch_acc.get_avg() * 100)

        self.epoch_acc.reset()
        self.epoch_loss.reset()

    def get_stats(self):
        return self.fit_res

    def get_epoch_info_str(self, is_train):
        if is_train:
            name = "train"
            loss = self.fit_res.train_loss[-1]
            acc = self.fit_res.train_acc[-1]
        else:
            name = "valid"
            loss = self.fit_res.test_loss[-1]
            acc = self.fit_res.test_acc[-1]

        return ' | {} loss {:5.2f} | {} acc {:8.2f}'.format(name, loss, name, acc)
