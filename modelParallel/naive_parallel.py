import torch
import torch.nn as nn
import warnings
from torch.cuda._utils import _get_device_index
from .shard_model import build_shards


class NaiveParallel(nn.Module):

    def __init__(self, module, device_ids=None, DEBUG=True):
        super(NaiveParallel, self).__init__()
        # ensure cuda is available
        if not torch.cuda.is_available():
            self.shards = nn.ModuleList(modules=[module])
            self.device_ids = ["cpu"]
            warnings.warn(
                "modelParallel should be invoked with gpus available")
            return

        # if no devices specified use all available gpus
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if DEBUG:
            device_ids.append("cpu")

        self.device_ids = list(
            map(lambda x: _get_device_index(x, True), device_ids))

        # shard the model
        self.shards = build_shards(module, len(self.device_ids))

        if len(self.device_ids) == 1:
            warnings.warn("modelParallel invoked with only 1 gpu available")

        # move shards to gpus
        for shard, device in zip(self.shards, self.device_ids):
            shard.to(device, non_blocking=True)

    def forward(self, x):
        for shard, device in zip(self.shards, self.device_ids):
            x = x.to(device)
            x = shard(x)
        return x
