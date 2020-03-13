import abc
from typing import Dict
from types import SimpleNamespace

# TODO: support for every X batches


class Stats(abc.ABC):
    """ Class to handle statistics collection """

    FIT_RESULTS_CLASS = SimpleNamespace

    def __init__(self):
        self.training = True
        self.fit_res = self.FIT_RESULTS_CLASS(num_epochs=0)
        assert not (self.fit_res is None)
        self.stats_config = dict()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_epoch_info_str(self, is_train):
        return ''

    def update_statistic_after_batch_single(self, name, value, n):
        """ NOTE: Attempt to replace the old function """
        cfg = self.stats_config[name]
        if cfg['per_epoch']:
            meter = getattr(self, f"epoch_{name}_meter")
            if not (value is None):
                meter.update(value, n=n)
            else:
                print(f"-W- NONE VALUE for {name}, val: {value}")

    def update_statistic_after_batch_all(self, d):
        for name, (value, n) in d.items():
            self.update_statistic_after_batch_single(name, value, n)

    def update_fit_res_after_batch_all(self, d):
        for name, value in d.items():
            if self.stats_config[name]['per_batch']:
                self.update_fit_res_after_batch_single(name, value[0])

    def update_fit_res_after_batch_single(self, name, value):
        cfg = self.stats_config[name]
        if cfg['per_batch']:
            self._append_value_to_fit_res_by_name(name, value, is_batch=True)

    def update_fit_res_after_epoch_all(self):
        list_names = [
            cfg for cfg, v in self.stats_config.items() if v['per_epoch']
        ]
        for name in list_names:
            self.update_fit_res_after_epoch_single(name)

    def update_fit_res_after_epoch_single(self, name):
        cfg = self.stats_config[name]
        if cfg['per_epoch']:
            meter = getattr(self, f"epoch_{name}_meter")
            value = meter.get_avg()
            self._append_value_to_fit_res_by_name(name, value, is_batch=False)
            meter.reset()

    def _append_value_to_fit_res_by_name(self, name, value, is_batch):
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

        if per_batch and per_epoch:
            raise NotImplementedError(
                "Statistics are supported for either batch or epoch.")

        setattr(self, f"epoch_{name}_meter", meter)
        fit_res_dict = []
        if train and test:
            # TODO: List[float]
            fit_res_dict.append(f"train_{name}")
            fit_res_dict.append(f"test_{name}")
        elif (train and not test) or (test and not train):
            fit_res_dict.append(f"{name}")
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

        for i in fit_res_dict:
            setattr(self.fit_res, i, [])

    @abc.abstractmethod
    def last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_batch_end(self, *args, **kw):
        pass

    def non_last_partition_on_epoch_end(self):
        pass

    def last_partition_on_epoch_end(self):
        if self.training:
            self.fit_res.num_epochs += 1

        self.update_fit_res_after_epoch_all()

    @abc.abstractmethod
    def get_stats(self, *args) -> Dict:
        pass
