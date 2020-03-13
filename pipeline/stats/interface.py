import abc
from typing import Dict


class Stats(abc.ABC):
    """ Class to handle statistics collection """

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @abc.abstractmethod
    def last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_epoch_end(self):
        pass

    @abc.abstractmethod
    def on_epoch_end(self):
        pass

    @abc.abstractmethod
    def get_stats(self, *args) -> Dict:
        pass

    def get_epoch_info_str(self, is_train):
        return ''

    def update_statistic_after_batch(self, name, value):
        if hasattr(self, f"epoch_{name}_meter"):
            meter = getattr(self, f"epoch_{name}_meter")
            if not (value is None):
                meter.update(value)
                # print(f"update_statistic_after_batch for {name}, val: {value}")
            else:
                print(f"-W- NONE VALUE for {name}, val: {value}")

        else:
            raise NotImplementedError(name)

    def has_statistic(self, name):
        return hasattr(self, f"epoch_{name}_meter")

    def add_statistic(self, name, meter):
        setattr(self, f"epoch_{name}_meter", meter)
