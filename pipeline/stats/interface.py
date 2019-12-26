import abc
from typing import Dict


class Stats(abc.ABC):
    """ Class to handle statistics collection """

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @abc.abstractmethod
    def on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_latst_partition_on_epoch_end(self):
        pass

    @abc.abstractmethod
    def on_epoch_end(self):
        pass

    @abc.abstractmethod
    def get_stats(self, *args) -> Dict:
        pass

    def get_epoch_info_str(self, is_train):
        return ''
