import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Union

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

__all__ = ['Partition', 'LastPartition']

class PartitionRngStasher:
    """ Utility class to stash and restore RNG state"""

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.state = {}

        # devices list for `fork_rng` method
        self.devices = [self.device] if self.device.type == 'cuda' else None

    def stash_rng_state(self, micro_batch_index):
        """ Stash RNG state """
        cpu_rng_state = torch.get_rng_state()
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                gpu_rng_state = torch.cuda.get_rng_state()
        else:
            gpu_rng_state = None

        self.state[micro_batch_index] = (cpu_rng_state, gpu_rng_state)

    def restore_rng_state(self, micro_batch_index):
        cpu_rng_state, gpu_rng_state = self.state[micro_batch_index]
        torch.set_rng_state(cpu_rng_state)
        if not (gpu_rng_state is None):
            torch.cuda.set_rng_state(gpu_rng_state, self.device)


class Partition(nn.Module):

    def __init__(self, layers, device):
        """
        :param layers: list of layers (or a single module)
        :param device: device of the partition
        """
        super(Partition, self).__init__()
        self.device = device
        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)  # .to(self.device)
        elif isinstance(layers, nn.Module):
            self.layers = layers

        self.cache_input = {}
        self.rng_stasher = PartitionRngStasher(device=self.device)
        self.to(self.device)

    def on_new_batch(self, num_micro_batches):
        # Create placeholder for micro batches input
        self.cache_input = {idx: None for idx in range(num_micro_batches)}
        self.stash_rngs = {idx: PartitionRngStasher(
            device=self.device) for idx in range(num_micro_batches)}

    def forward(self, x: TensorOrTensors, micro_batch_idx):

        if self.training:  # Dummy fwd to save input and pass output to next layer
            # do the dummy forward
            # stash rng state
            # save input for later (recomputation)
            # TODO: can spare the detach
            self.rng_stasher.stash_rng_state(micro_batch_idx)

            with torch.no_grad():
                if isinstance(x, Tensor):
                    x = x.detach().to(self.device).requires_grad_()
                    self.cache_input[micro_batch_idx] = x
                    x = self.layers(x)
                else:
                    # FIXME: explicitly creating new objects
                    x = [y.detach().to(self.device).requires_grad_()
                         for y in x]
                    self.cache_input[micro_batch_idx] = x
                    x = self.layers(*x)
            return x
        else:
            with torch.no_grad():
                if isinstance(x, Tensor):
                    x = x.to(self.device)
                    x = self.layers(x)
                else:
                    x = [y.to(self.device) for y in x]
                    x = self.layers(*x)
                return x

    def recompute_and_backward(self, g, micro_batch_idx):
        with torch.random.fork_rng(devices=self.rng_stasher.devices):
            self.rng_stasher.restore_rng_state(micro_batch_idx)
            x = self.cache_input[micro_batch_idx]
            if isinstance(x, Tensor):
                x = self.layers(x)
                torch.autograd.backward(x, g)
            else:
                x = self.layers(*x)
                torch.autograd.backward(x, g)

    def get_grad(self, micro_batch_idx):
        x = self.cache_input[micro_batch_idx]
        if isinstance(x, Tensor):
            return x.grad
        else:
            return [y.grad for y in x]

    def backward(self, g, **kw):
        raise NotImplementedError()


class LastPartition(Partition):
    # TODO: make the inheritance true subtype.

    def __init__(self, layers, device):
        super(LastPartition, self).__init__(layers, device)

    def forward(self, x, micro_batch_idx):

        if self.training:
            # Note that here we save the input just to get its gradeint later
            # we do not plan to do any recomputation.

            if isinstance(x, Tensor):
                x = x.detach().to(self.device).requires_grad_()
                self.cache_input[micro_batch_idx] = x
                x = self.layers(x)
            else:
                x = [y.detach().to(self.device).requires_grad_()
                     for y in x]  # FIXME: explicitly creating new objects
                self.cache_input[micro_batch_idx] = x
                x = self.layers(*x)

            return x
        else:
            with torch.no_grad():
                if isinstance(x, Tensor):
                    x = x.to(self.device)
                    x = self.layers(x)
                else:
                    x = [y.to(self.device) for y in x]
                    x = self.layers(*x)

                    # Last partition outputs results in a non tensor format
                    if not isinstance(x, Tensor):
                        assert(len(x) == 1)
                        x = x[0]
            return x

    def recompute_and_backward(self, *args):
        raise NotImplementedError()


class GpipePartition:
    def __init__(self, layers, device, recomputation=True):
        """
        :param layers: list of layers (or a single layer)
        :param device: device of the partition
        """
        super(GpipePartition, self).__init__()
        self.device = device
        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)  # .to(self.device)
        elif isinstance(layers, nn.Module):
            self.layers = layers
        
        self.recomputation = recomputation
        if self.recomputation:
            self.cache_input = {}
            self.rng_stasher = PartitionRngStasher(device=self.device)
        self.to(self.device)

    def on_new_batch(self, num_micro_batches):
        if not self.recomputation:
            return
        # Create placeholder for micro batches input
        self.cache_input = {idx: None for idx in range(num_micro_batches)}
        self.stash_rngs = {idx: PartitionRngStasher(
            device=self.device) for idx in range(num_micro_batches)}

    def forward(self, x: TensorOrTensors, micro_batch_idx):
        # TODO
        pass
