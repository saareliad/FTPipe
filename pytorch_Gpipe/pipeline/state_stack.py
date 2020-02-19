from collections import deque
from typing import List

import torch
from torch import Tensor

from .utils import InvalidState


class StateStack():
    ''' A stack managing the saved activations and rng state of the partition
    '''

    def __init__(self, device: torch.device):
        self.device = device
        self.states = deque()
        self.activations = deque()

    def push(self, xs: List[Tensor], save_state: bool = False) -> List[Tensor]:
        xs = [x.data.clone() for x in xs]
        self.activations.append(xs)
        if save_state:
            if self.device.type == 'cpu':
                self.states.appendleft(torch.get_rng_state())
            else:
                self.states.appendleft(torch.cuda.get_rng_state(self.device))
        return xs

    def pop(self, remove_state: bool = False) -> List[Tensor]:
        if len(self.activations) == 0 or len(self.states) == 0:
            raise InvalidState("cannot restore activation as none are saved")
        activations = self.activations.pop()
        activations = [t.requires_grad_() for t in activations]
        if self.device.type == 'cpu':
            torch.set_rng_state(self.states[-1])
        else:
            torch.cuda.set_rng_state(self.states[-1], device=self.device)
        if remove_state:
            self.states.pop()
        return activations
