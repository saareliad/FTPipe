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
        self.stats_config = dict()

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

    def update_statistic_after_batch_single(self, name, value):
        """ NOTE: Attempt to replace the old function """
        cfg = self.stats_config[name]
        if cfg['per_epoch']:
            meter = getattr(self, f"epoch_{name}_meter")
            if not (value is None):
                meter.update(value)
            else:
                print(f"-W- NONE VALUE for {name}, val: {value}")

    def update_statistic_after_batch_all(self, d):
        for name, value in d.items():
            self.update_statistic_after_batch_single(name, value)

    def update_fit_res_after_batch_all(self, d):
        for name, value in d.items():
            self.update_fit_res_after_batch_single(name, value)

    def update_fit_res_after_batch_single(self, name, value):
        cfg = self.stats_config[name]
        if cfg['per_batch']:
            self._append_value_to_fit_res_by_name(name, value)

    def update_fit_res_after_epoch_all(self):
        list_names = [
            cfg for cfg, v in self.stats_config.items() if v['per_epoch']
        ]
        for name in list_names:
            self.update_fit_rest_after_epoch_single(name)

    def update_fit_rest_after_epoch_single(self, name):
        cfg = self.stats_config[name]
        if cfg['per_epoch']:
            meter = getattr(self, f"epoch_{name}_meter")
            value = meter.get_avg()
            self._append_value_to_fit_res_by_name(name, value)
            meter.reset()

    def _append_value_to_fit_res_by_name(self, name, value):
        cfg = self.stats_config[name]
        if (self.training and not cfg['train']) or (not self.training
                                                    and not cfg['test']):
            return

        # Get fit name
        if cfg['train'] and cfg['test']:
            fit_name = f"train_{name}" if self.training else f"test_{name}"
        else:
            fit_name = name

        fit_stat = getattr(self.fit_res, fit_name)
        fit_stat.append(value)

    def has_statistic(self, name):
        return hasattr(self, f"epoch_{name}_meter")

    def add_statistic(self,
                      name,
                      meter,
                      per_batch=False,
                      per_epoch=True,
                      train=True,
                      test=True):
        setattr(self, f"epoch_{name}_meter", meter)

        # setattr(self, f"record_{name}_per_batch", per_batch)

        if per_batch and per_epoch:
              # TODO: because currently they have same names...
            raise NotImplementedError()

        fit_res_dict = set()
        if train and test:
            # TODO: List[float]
            fit_res_dict.add(f"train_{name}")
            fit_res_dict.add(f"test_{name}")
        elif (train and not test) or (test and not train):
            fit_res_dict.add(f"{name}")
        else:
            raise ValueError()

        self.stats_config[name] = {
            "train": train,
            "test": test,
            "per_batch": per_batch,
            "per_epoch": per_epoch,
            # "meter": meter,
            "fit_res": fit_res_dict
        }

    @abc.abstractmethod
    def fit_result_init_dict(self):
        # TODO: do this automatically
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
