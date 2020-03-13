import abc
from typing import Dict
from types import SimpleNamespace


class Stats(abc.ABC):
    """ Class to handle statistics collection """

    FIT_RESULTS_CLASS = SimpleNamespace

    def __init__(self):
        self.training = True
        self.fit_res = self.FIT_RESULTS_CLASS(**self.fit_result_init_dict())
        assert not (self.fit_res is None)
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_epoch_info_str(self, is_train):
        return ''

    def update_statistic_after_batch(self, name, value):
        """ Updating epoch statistics meter after batch """
        if hasattr(self, f"epoch_{name}_meter"):
            meter = getattr(self, f"epoch_{name}_meter")
            if not (value is None):
                meter.update(value)
            else:
                print(f"-W- NONE VALUE for {name}, val: {value}")
        else:
            raise NotImplementedError(name)

    def update_fit_res_after_epoch(self, name, value):
        raise NotImplementedError()


    def foo():
        pass
    #     if self.record_loss_per_batch:
    # if self.training:
    #     self.fit_res.train_loss.append(loss)
    #     self.fit_res.train_ppl.append(math.exp(loss))
    # else:
    #     self.fit_res.test_loss.append(loss)
    #     self.fit_res.test_ppl.append(math.exp(loss))

    def has_statistic(self, name):
        return hasattr(self, f"epoch_{name}_meter")

    def add_statistic(self, name, meter):
        setattr(self, f"epoch_{name}_meter", meter)

    @abc.abstractmethod
    def fit_result_init_dict(self):
        pass

    @abc.abstractmethod
    def last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_epoch_end(self):
        pass

    @abc.abstractmethod
    def last_partition_on_epoch_end(self):
        pass

    @abc.abstractmethod
    def get_stats(self, *args) -> Dict:
        pass

