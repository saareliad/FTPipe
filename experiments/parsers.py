import sys
sys.path.append("../")
import argparse
import torch
import sample_models
import inspect
from importlib import import_module


class StoreDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kv = {}
        if not isinstance(values, (list,)):
            values = (values,)
        for value in values:
            n, v = value.split('=')
            try:
                kv[n] = int(v)
            except ValueError:
                kv[n] = v
        setattr(namespace, self.dest, kv)


class ExpParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.available_models = self._get_available_models()
        self.add_argument('--model', '-m',
                          help='The sample model we want to run the experiment on.\n for example alexnet in order to use sample_models.alexnet',
                          choices=self.available_models.keys(), required=True, dest='model_class')
        self.add_argument('--pipeline_config', '-p',
                          help="module of pipeline config")
        self.add_argument('--model_args',
                          help='The parameters for the model\nkeys are always string and values can be ints or strings with int taking precedence',
                          nargs='*', action=StoreDict,
                          default=dict())
        self.add_argument('--num_micro_batches',
                          help='number of microbatches to split the input to',
                          default=0)
        self.add_argument('--device_ids', '-d', help='device ids to use',
                          type=int, nargs='*', default=[])
        self.add_argument('--no_dp', default=False,
                          help="whether to not compare with data parallel", action="store_true")
        self.add_argument('--no_single', default=False,
                          help="whether to not compare with single gpu", action="store_true")

    def parse_args(self, *args, **kwargs):
        res = vars(super().parse_args(*args, **kwargs))

        res['model_class'] = self.available_models[res['model_class']]

        device_ids = self.parse_devices(res['device_ids'])

        config_module = import_module(res['pipeline_config'])

        n_parts = sum(1 for _, m in inspect.getmembers(config_module)
                      if isinstance(m, type))

        assert len(device_ids) >= n_parts,\
            f"expected {n_parts} gpus got {len(device_ids)}"

        res['device_ids'] = device_ids[:n_parts]

        res['pipeline_config'] = config_module.create_pipeline_configuration

        n = min(n_parts, res['num_micro_batches'])

        if n == 0:
            n = n_parts

        res['num_micro_batches'] = n

        return res

    def parse_devices(self, devices):
        if not devices:
            devices = range(torch.cuda.device_count())

        return [torch.cuda.device(idx) for idx in devices]

    def _get_available_models(self):
        classes = dict()
        for name, t in inspect.getmembers(sample_models):
            if inspect.isfunction(t):
                classes[name] = t

        return classes


class BatchInfoExpParser(ExpParser):
    def __init__(self, *args, **kwargs):
        super(BatchInfoExpParser, self).__init__(*args, **kwargs)
        self.add_argument("--gpu_batch_size", type=int, required=True,
                          help="batch_size to use. for dataParallel will be divided by the number of gpus", default=1)
        self.add_argument('--batch_shape', type=int, nargs="+", default=(3, 224, 224),
                          help="batch shape to use without batch idx")
        self.add_argument('--batch_dim', type=int, default=0,
                          help="the batch dim of the sample")


class ImageNetParser(argparse.ArgumentParser):
    def __init__(self, available_models, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.available_models = available_models
        self.add_argument(
            "--epochs", help="number of train epochs", default=10)
        self.add_argument("--data_path",
                          help='path to imagenet folder', default="./imagenet")
        self.add_argument("--batch_size",
                          help="train/validation batch size", type=int, default=256)
        self.add_argument("--num_data_workers",
                          help="number of dataloading workers", default=4)
        self.add_argument("--lr",
                          help="learning rate for SGD optimizer", default=0.1, type=float)
        self.add_argument("--momentum", help="momentum for SGD optimizer",
                          default=0.9, type=float)
        self.add_argument("--weight_decay", help="weight decay for SGD optimizer",
                          default=1e-4, type=float)
        self.add_argument("--nesterov", help="whether to use nesterov with SGD optimizer",
                          default=False, action="store_true")

        self.add_argument("--dp", help="train dataParallel model",
                          action="store_true", default=False)
        self.add_argument("--single", help="train single gpu model",
                          action="store_true", default=False)
        self.add_argument("--pipeline", help="train pipeline model",
                          action="store_true", default=False)

        self.add_argument('--pipeline_config', '-p',
                          help="module of pipeline config", default="")
        self.add_argument('--num_micro_batches',
                          help='number of microbatches to split the input to',
                          default=0)

        self.add_argument('--model', '-m',
                          help='The sample model we want to run the experiment on.\n for example alexnet in order to use sample_models.alexnet',
                          choices=self.available_models.keys(), required=True, dest='model_class')
        self.add_argument('--model_args',
                          help='The parameters for the model\nkeys are always string and values can be ints or strings with int taking precedence',
                          nargs='*', action=StoreDict,
                          default=dict())

        self.add_argument('--device_ids', '-d', help='device ids to use',
                          type=int, nargs='*', default=[])

    def parse_args(self, *args, **kwargs):
        res = vars(super().parse_args(*args, **kwargs))

        res['model_class'] = self.available_models[res['model_class']]

        device_ids = self.parse_devices(res['device_ids'])

        config_module = import_module(res['pipeline_config'])

        n_parts = sum(1 for _, m in inspect.getmembers(config_module)
                      if isinstance(m, type))

        assert len(device_ids) >= n_parts,\
            f"expected {n_parts} gpus got {len(device_ids)}"

        res['device_ids'] = device_ids[:n_parts]

        res['pipeline_config'] = config_module.create_pipeline_configuration

        n = min(n_parts, res['num_micro_batches'])

        if n == 0:
            n = n_parts

        res['num_micro_batches'] = n

        return res

    def parse_devices(self, devices):
        if not devices:
            devices = range(torch.cuda.device_count())

        return [torch.cuda.device(idx) for idx in devices]
