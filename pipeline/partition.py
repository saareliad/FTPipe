import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union
from .monkey_patch import DummyForwardMonkeyPatcher
from .replace_inplace import replace_inplace_for_first_innermost_layer_
# import logging
Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

__all__ = ['Partition', 'LastPartition', 'FirstPartition']

DEFAULT_CLASSES_LIST_TO_PATCH = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                 nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]


class PartitionRngStasher:
    """
    Utility class to stash and restore RNG state
    pop happens when re restore the state (therefore we can only restore once).
    """

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.state = {}

        # devices list for `fork_rng` method
        self.devices = [self.device] if self.device.type == 'cuda' else []

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
        cpu_rng_state, gpu_rng_state = self.state.pop(micro_batch_index)
        torch.set_rng_state(cpu_rng_state)
        if not (gpu_rng_state is None):
            torch.cuda.set_rng_state(gpu_rng_state, self.device)

    def clear_state(self):
        self.state.clear()


class Partition(nn.Module):
    """
    Partition with recomputation.

    saves activations.
    pop happens when we read the gradient.
    """
    _REQ_GRAD = True
    _HAS_DUMMY_FORWARD = True

    def __init__(self, layers, device, to_device=True, classes_list_to_patch=DEFAULT_CLASSES_LIST_TO_PATCH):
        """
        :param layers: list of layers (or a single module)
        :param device: device of the partition
        """
        super(Partition, self).__init__()
        self.device = device
        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)
        elif isinstance(layers, nn.Module):
            self.layers = layers

        if self._HAS_DUMMY_FORWARD:
            # TODO: can print if is_replaced
            replace_inplace_for_first_innermost_layer_(self.layers)

        self.dummy_forward_monkey_patcher = DummyForwardMonkeyPatcher(self.layers, classes_list_to_patch) \
            if self._HAS_DUMMY_FORWARD else None
        self.input_buffer = {}
        self.rng_stasher = PartitionRngStasher(device=self.device)

        if to_device:
            self.to(self.device)

    def on_new_batch(self, num_micro_batches):
        # Create placeholder for micro batches input and rng states.
        self.input_buffer = {idx: None for idx in range(num_micro_batches)}

    def forward(self, x: TensorOrTensors, micro_batch_idx):

        if self.training:  # Dummy fwd to save input and pass output to next layer
            # do the dummy forward
            # stash rng state
            # save input for later (recomputation)
            # TODO: can spare the detach

            if self.dummy_forward_monkey_patcher:
                self.dummy_forward_monkey_patcher.sync()
                self.dummy_forward_monkey_patcher.replace_for_dummy()

            with torch.no_grad():
                # EXPLICITLY DO CLONE
                if isinstance(x, Tensor):
                    # Note - we clone here because we don't want the tensor to get overriden.
                    # TODO: it could be done better if we use multiple input buffers instead of allocating
                    # In pytorch it can happen auto matically with THCCashingAlocator.
                    x = x.data.clone().requires_grad_(self._REQ_GRAD)
                    self.input_buffer[micro_batch_idx] = x
                    self.rng_stasher.stash_rng_state(micro_batch_idx)
                    x = self.layers(x)
                else:
                    x = [tensor.data.clone().requires_grad_(self._REQ_GRAD)
                         for tensor in x]
                    self.input_buffer[micro_batch_idx] = x
                    self.rng_stasher.stash_rng_state(micro_batch_idx)
                    x = self.layers(*x)

                if self.dummy_forward_monkey_patcher:
                    self.dummy_forward_monkey_patcher.replace_for_forward()
            return x

        else:
            with torch.no_grad():
                if self.dummy_forward_monkey_patcher:
                    self.dummy_forward_monkey_patcher.replace_for_forward()
                if isinstance(x, Tensor):
                    # x = x.to(self.device)
                    x = self.layers(x)
                else:
                    # raise NotImplementedError()
                    x = [y for y in x]
                    x = self.layers(*x)
                return x

    def recompute_and_backward(self, g, micro_batch_idx):
        # TODO: can make these two functions (recompute, backwards)
        # To enable scheduling the recompute
        x = self.input_buffer[micro_batch_idx]  # Note: still not poping!
        if self.dummy_forward_monkey_patcher:
            self.dummy_forward_monkey_patcher.replace_for_forward()
        # TODO: maybe changing the rng state messes up with MPI?
        with torch.random.fork_rng(devices=self.rng_stasher.devices):
            self.rng_stasher.restore_rng_state(micro_batch_idx)
            if isinstance(x, Tensor):
                x = self.layers(x)
                # logging.getLogger("msnag").info(f"device:{self.layers.__class__.__name__[-1]} max x:{[z.data.max() for z in x]}, max grad:{[z.max() for z in g]}")
                # for p in self.parameters():
                #     print(p.abs().max())
            else:
                # raise NotImplementedError()
                x = self.layers(*x)
        torch.autograd.backward(x, g)

    def get_grad(self, micro_batch_idx):
        x = self.input_buffer.pop(micro_batch_idx)
        if isinstance(x, Tensor):
            # logging.getLogger("msnag").info(f"device:{self.layers.__class__.__name__[-1]} max_grad_norm:{x.grad.norm()}")
            return x.grad.data
        else:
            return [y.grad.data for y in x]

    def backward(self, g, **kw):
        raise NotImplementedError()


class FirstPartition(Partition):
    """ The first partition does not need to record gradients of stashed inputs.
        This may save some memory.
    """
    _REQ_GRAD = False
    _HAS_DUMMY_FORWARD = True

    def __init__(self, *args, **kw):
        super(FirstPartition, self).__init__(*args, **kw)

    def recompute_and_backward(self, g, micro_batch_idx):
        # Unlike normal partition, here we pop the activations when we read from buffer
        x = self.input_buffer.pop(micro_batch_idx)  # Note: here we pop.
        if self.dummy_forward_monkey_patcher:
            self.dummy_forward_monkey_patcher.replace_for_forward()
        with torch.random.fork_rng(devices=self.rng_stasher.devices):
            self.rng_stasher.restore_rng_state(micro_batch_idx)
            if isinstance(x, Tensor):
                x = self.layers(x)
                # logging.getLogger("msnag").info(f"device:{self.layers.__class__.__name__[-1]} max x:{[z.data.max() for z in x]}, max grad:{[z.max() for z in g]}")
                # for p in self.parameters():
                #     print(p.abs().max())
            else:
                # raise NotImplementedError()
                x = self.layers(*x)
        torch.autograd.backward(x, g)

    def get_grad(self, micro_batch_idx):
        return None


class LastPartition(Partition):
    _REQ_GRAD = True
    _HAS_DUMMY_FORWARD = False
    # TODO: make the inheritance true subtype.

    def __init__(self, *args, **kw):
        super(LastPartition, self).__init__(*args, **kw)

    def forward(self, x, micro_batch_idx):
        if self.training:
            # Note that here we save the input just to get its gradeint later
            # we do not plan to do any recomputation.

            # Note: we don't copy the tensor here to save memory,
            #     # we don't care that the next recv will override it,
            #     # as all we need from it is its grad, imidaitly after.
            #     # (otherwise, we have to do synchrounous recvs)
            # TODO: can we avoid the detach?

            if isinstance(x, Tensor):
                # # See note on option 1 below.
                with torch.no_grad():
                    x = x.detach_().requires_grad_()
                self.input_buffer[micro_batch_idx] = x
                x = self.layers(x)
            else:
                # Option 2
                with torch.no_grad():
                    x = [tensor.detach_().requires_grad_() for tensor in x]

                self.input_buffer[micro_batch_idx] = x
                x = self.layers(*x)
        else:
            with torch.no_grad():
                if isinstance(x, Tensor):
                    x = self.layers(x.data)
                else:
                    x = [y.data for y in x]
                    x = self.layers(*x)

        #  Last partition outputs results in a non tensor format
        if not isinstance(x, Tensor):
            assert(len(x) == 1)
            return x[0]
        return x

    def recompute_and_backward(self, *args):
        raise NotImplementedError()


##################################################
# Unrelated but still here, may be useful later
##################################################


# class GpipePartition:
#     """ TODO: uncompleted version of GpipePartition.... """

#     def __init__(self, layers, device, recomputation=True):
#         """
#         :param layers: list of layers (or a single layer)
#         :param device: device of the partition
#         """
#         super(GpipePartition, self).__init__()
#         self.device = device
#         if isinstance(layers, list):
#             self.layers = nn.Sequential(*layers)  # .to(self.device)
#         elif isinstance(layers, nn.Module):
#             self.layers = layers

#         self.recomputation = recomputation
#         if self.recomputation:
#             self.input_buffer = {}
#             self.rng_stasher = PartitionRngStasher(device=self.device)
#         self.to(self.device)

#     def on_new_batch(self, num_micro_batches):
#         if not self.recomputation:
#             return
#         # Create placeholder for micro batches input
#         self.input_buffer = {idx: None for idx in range(num_micro_batches)}

#     def forward(self, x: TensorOrTensors, micro_batch_idx):
#         # TODO
#         pass
