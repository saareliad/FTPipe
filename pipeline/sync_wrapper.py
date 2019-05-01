import torch
import torch.nn as nn
import torch.distributed as dist


class SyncWrapper(nn.Module):
    def __init__(self, module: nn.Module, device: str):
        super(SyncWrapper, self).__init__()
        self.module = module
        self.device = device

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dist.barrier()
        input = input.to(self.device)
        # input.register_hook(dist.barrier)
        return self.module(input)

