import torch
import torch.nn as nn
from pipeline.utils import AutoResetBarrier


class SyncWrapper(nn.Module):
    def __init__(self, module: nn.Module, device: str, barrier: AutoResetBarrier = None):
        super(SyncWrapper, self).__init__()
        self.module = module
        self.device = device
        self.barrier = barrier

    def set_barrier(self, barrier: AutoResetBarrier):
        self.barrier = barrier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.barrier.wait()
        input = input.to(self.device)
        # input.register_hook(dist.barrier)
        return self.module(input)

