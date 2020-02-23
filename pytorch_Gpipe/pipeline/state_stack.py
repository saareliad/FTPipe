from collections import deque
from typing import List

import torch
from torch import Tensor


class StateStack():
    ''' A stack managing the saved activations and rng state of the partition
    '''

    def __init__(self, device: torch.device):
        self.device = device
        self.states = deque()
        self.activations = deque()

    def save_rng_state(self):
        cpu_state = torch.get_rng_state()
        if self.device.type == 'cuda':
            gpu_state = torch.cuda.get_rng_state(self.device)
        else:
            gpu_state = None

        self.states.append((cpu_state, gpu_state))

    def restore_rng_state(self):
        cpu_state, gpu_state = self.states.popleft()
        torch.set_rng_state(cpu_state)
        if not (gpu_state is None):
            torch.cuda.set_rng_state(gpu_state, self.device)

    def save_activation(self, xs: List[Tensor]) -> List[Tensor]:
        xs = [x.data.clone() for x in xs]
        self.activations.append(xs)
        return xs

    def restore_activation(self) -> List[Tensor]:
        activations = self.activations.popleft()
        activations = [t.requires_grad_() for t in activations]
        return activations
