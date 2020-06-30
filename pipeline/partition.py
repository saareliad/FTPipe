import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union
from .monkey_patch import DummyForwardMonkeyPatcher
from .monkey_patch.find_modules import find_modules
from .replace_inplace import replace_inplace_for_first_innermost_layer_
from .rng_stasher import PartitionRngStasher
from . import dp_sim
import types
from functools import partial
from .util import flatten, unflatten, nested_map

# import logging
Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

__all__ = [
    # Async pipe
    'Partition',
    'FirstPartition',
    'LastPartition',
    'LastPartitionWithLabelInput',
    'PartitionWithoutRecomputation',
    'get_buffers_for_ddp_sync',
    # GPipe:
    'GPipePartition',
    'GPipeFirstPartition',
    'GPipeLastPartition',
    'GPipeLastPartitionWithLabelInput'
]

# TODO: LayerNorm? GroupNorm?
DEFAULT_CLASSES_LIST_TO_PATCH = [
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    dp_sim.BatchNorm1d, dp_sim.BatchNorm2d, dp_sim.BatchNorm3d
]


def get_buffers_for_ddp_sync(model,
                             classes_to_patch=DEFAULT_CLASSES_LIST_TO_PATCH):
    # This function should be used once.

    for model_to_patch in classes_to_patch:
        found = []  # list of tuples: (access_string, model)
        find_modules(model, "", model_to_patch, found)

    found = sorted(found, key=lambda t: t[0])
    buffers = []
    for (access_string, model) in found:
        buffers.extend(sorted(model.named_buffers(), key=lambda t: t[0]))

    buffers = [t[1] for t in buffers]
    return buffers


class Partition(nn.Module):
    """
    Partition with recomputation.
    Should be used as Intermidiate partition.

    saves activations.
    pop happens when we read the gradient.

    NOTE: there are other class for LastPartition and FirstPartition, to be used as needed.
    """
    _REQ_GRAD = True
    _HAS_DUMMY_FORWARD = True
    _CLONE_INPUTS = True

    def __init__(self,
                 layers,
                 device,
                 to_device=True,
                 classes_list_to_patch=DEFAULT_CLASSES_LIST_TO_PATCH,
                 req_grad=None,
                 outputs_req_grad=None):
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

        if self._HAS_DUMMY_FORWARD or self._REQ_GRAD:
            # TODO: can print if is_replaced
            replace_inplace_for_first_innermost_layer_(self.layers)

        self.dummy_forward_monkey_patcher = DummyForwardMonkeyPatcher(self.layers, classes_list_to_patch) \
            if self._HAS_DUMMY_FORWARD else None

        # just make it None if nothing to patch.
        if self.dummy_forward_monkey_patcher is not None and not self.dummy_forward_monkey_patcher.models:
            self.dummy_forward_monkey_patcher = None

        # input_buffer is for saving activations.
        # we save activations for:
        #   (1) back-propagate thier gradients
        #   (2) recomputation
        self.input_buffer = {}
        self.bwd_graph_head_buffer = {}  # For recompute
        self.rng_stasher = PartitionRngStasher(device=self.device)
        self.req_grad = req_grad_dict_to_tuple(req_grad)
        self.outputs_req_grad = req_grad_dict_to_tuple(outputs_req_grad)
        if to_device:
            self.to(self.device)

    def forward(self, x: TensorOrTensors, micro_batch_idx):

        if self.training:  # Dummy fwd to save input and pass output to next layer
            # do the dummy forward
            # stash rng state
            # save input for later (recomputation)
            if self.dummy_forward_monkey_patcher:
                self.dummy_forward_monkey_patcher.sync()
                self.dummy_forward_monkey_patcher.replace_for_dummy()

            with torch.no_grad():

                if isinstance(x, Tensor):
                    # EXPLICITLY DO CLONE
                    if self._CLONE_INPUTS:
                        x = x.detach().clone().requires_grad_(self.req_grad[0])
                    else:
                        x = x.detach().requires_grad_(self.req_grad[0])
                    self.input_buffer[micro_batch_idx] = x
                    self.rng_stasher.stash_rng_state(micro_batch_idx)
                    x = self.layers(x)

                else:
                    # EXPLICITLY DO CLONE
                    if self._CLONE_INPUTS:
                        x = list(get_dcr(x, self.req_grad))
                    else:
                        x = list(get_dr(x, self.req_grad))
                    self.input_buffer[micro_batch_idx] = x
                    self.rng_stasher.stash_rng_state(micro_batch_idx)
                    # UNFLATTEN
                    x = unflatten(x, self.req_grad)
                    x = self.layers(*x)

                if self.dummy_forward_monkey_patcher:
                    self.dummy_forward_monkey_patcher.replace_for_forward()
            return x

        else:
            with torch.no_grad():
                if self.dummy_forward_monkey_patcher:
                    self.dummy_forward_monkey_patcher.replace_for_forward()

                if isinstance(x, Tensor):
                    x = self.layers(x)
                else:
                    # UNFLATTEN
                    x = unflatten(x, self.req_grad)
                    x = self.layers(*x)
                return x

    def recompute(self, micro_batch_idx):
        x = self.input_buffer[micro_batch_idx]  # Note: still not poping!
        if self.dummy_forward_monkey_patcher:
            self.dummy_forward_monkey_patcher.replace_for_forward()
        # TODO: maybe changing the rng state messes up with MPI?
        with torch.random.fork_rng(devices=self.rng_stasher.devices):
            self.rng_stasher.restore_rng_state(micro_batch_idx)

            #  UNFLATTEN
            x = unflatten(x, self.req_grad)
            x = self.layers(*x)
        # Save for later
        self.bwd_graph_head_buffer[micro_batch_idx] = x

    def backward_from_recomputed(self, g, micro_batch_idx):
        x = self.bwd_graph_head_buffer.pop(micro_batch_idx)
        x, g = filter_for_backward(x, g)
        torch.autograd.backward(x, g)

    def get_grad(self, micro_batch_idx):
        """ returns an iteretable of grads """
        x = self.input_buffer.pop(micro_batch_idx)  # Flattened
        if isinstance(x, Tensor):
            return (x.grad, )
        else:
            return [y.grad for y in filter_req_grad_tensors(x)]

    def backward(self, g, **kw):
        raise NotImplementedError()


class FirstPartition(Partition):
    """ The first partition does not need to record gradients of stashed inputs.
        This may save some memory.
        We don't clone inputs.
        We don't record gradients for inputs.
    """
    _REQ_GRAD = False
    _HAS_DUMMY_FORWARD = True

    def __init__(self, *args, **kw):
        super(FirstPartition, self).__init__(*args, **kw)

    def forward(self, x: TensorOrTensors, micro_batch_idx):

        if self.training:  # Dummy fwd to save input and pass output to next layer
            # do the dummy forward
            # stash rng state
            # save input for later (recomputation)
            if self.dummy_forward_monkey_patcher:
                self.dummy_forward_monkey_patcher.sync()
                self.dummy_forward_monkey_patcher.replace_for_dummy()

            with torch.no_grad():
                # EXPLICITLY DO CLONE
                if isinstance(x, Tensor):
                    # Note - we clone here because we don't want the tensor to get overriden.
                    # TODO: it could be done better if we use multiple input buffers instead of allocating
                    # (when #buffers==#max(len(input_buffer)))
                    # In pytorch it can happen auto matically with THCCashingAlocator.
                    self.input_buffer[micro_batch_idx] = x
                    self.rng_stasher.stash_rng_state(micro_batch_idx)
                    x = self.layers(x)
                else:
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
                    x = self.layers(x)
                else:
                    x = self.layers(*x)
                return x

    def recompute(self, micro_batch_idx):
        # Unlike normal partition, here we pop the activations after we read from buffer
        # This is a sperate function with code copy,
        # because we want to pop as early as possible to possibly save memory.
        x = self.input_buffer.pop(micro_batch_idx)  # Note: here we pop.

        # #### CODE COPY FROM super().recompute ########
        if self.dummy_forward_monkey_patcher:
            self.dummy_forward_monkey_patcher.replace_for_forward()
        with torch.random.fork_rng(devices=self.rng_stasher.devices):
            self.rng_stasher.restore_rng_state(micro_batch_idx)
            if isinstance(x, Tensor):
                x = self.layers(x)
            else:
                x = self.layers(*x)

        self.bwd_graph_head_buffer[micro_batch_idx] = x
        # #### CODE COPY FROM super().recompute ########

    def get_grad(self, micro_batch_idx):
        return NotImplementedError()


class LastPartition(Partition):
    _REQ_GRAD = True
    _HAS_DUMMY_FORWARD = False

    def __init__(self, *args, **kw):
        super(LastPartition, self).__init__(*args, **kw)

    def forward(self, x: TensorOrTensors, micro_batch_idx):
        if self.training:
            # Note that here we save the input just to get its gradeint later
            # we do not plan to do any recomputation.

            # Note: we don't copy the tensor here to save memory,
            #     # we don't care that the next recv will override it,
            #     # as all we need from it is its grad, imidaitly after.
            #     # (otherwise, we have to do synchrounous recvs)
            # TODO: can we avoid the detach?
            # NOTE: currently there is not detach here!

            # TODO: if _HAS_DUMMY_FORWARD == False we can store only activations which need gradients.
            if isinstance(x, Tensor):
                # # See note on option 1 below.
                x = x.requires_grad_(self.req_grad[0])
                self.input_buffer[micro_batch_idx] = x
                x = self.layers(x)
            else:
                # Option 2 
                x = list(get_r(x, self.req_grad))
                self.input_buffer[micro_batch_idx] = x

                # UNFLATEN
                x = unflatten(x, self.req_grad)
                x = self.layers(*x)
        else:
            with torch.no_grad():
                if isinstance(x, Tensor):
                    x = self.layers(x)
                else:

                    # UNFLATEN
                    x = unflatten(x, self.req_grad)
                    x = self.layers(*x)

        #  Last partition outputs should be in a tensor format
        if not isinstance(x, Tensor):
            assert (len(x) == 1)
            return x[0]
        return x


class LastPartitionWithLabelInput(LastPartition):
    """A assuming that given x is tuple in which last idx is the label.
        We don't store the label, because we don't need gradient on it.
        
        In use for our partitoned transformers with LMhead.
    """
    # TODO: 
    # _REQ_GRAD = True
    # _HAS_DUMMY_FORWARD = False
    def forward(self, x: TensorOrTensors, micro_batch_idx):
        assert not isinstance(x, Tensor)
        req_grad = self.req_grad
        if self.training:

            # For backprobpagating gradients
            x = list(get_r(x, req_grad))
            self.input_buffer[micro_batch_idx] = list(filter_req_grad_tensors(flatten(x)))

            # UNFLATEN
            x = unflatten(x, req_grad)
            x = self.layers(*x)
        else:
            with torch.no_grad():
                # UNFLATEN
                x = unflatten(x, req_grad)
                x = self.layers(*x)

        #  Last partition outputs should be in a tensor format
        if not isinstance(x, Tensor):
            assert (len(x) == 1)
            return x[0]
        return x


class PartitionWithoutRecomputation(nn.Module):
    # _REQ_GRAD = True
    _HAS_DUMMY_FORWARD = False
    _CLONE_INPUTS = True

    def __init__(self,
                 layers,
                 device,
                 to_device=True,
                 _REQ_GRAD=True,
                 req_grad=None,
                 outputs_req_grad=None):
        """
            Intermidiate partition which does not do recomputation.
            HACK: has misleading names to be used with existing code.

            NOTE:
                (1) This partition should (ideally) be accompanied by weight stashing for async pipeline, 
                but it also works without it.
                (2) use _REQ_GRAD=False for first partition
        """
        super().__init__()

        self.device = device
        self._REQ_GRAD = _REQ_GRAD
        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)
        elif isinstance(layers, nn.Module):
            self.layers = layers

        if self._REQ_GRAD or self._HAS_DUMMY_FORWARD:
            # TODO: can print if is_replaced
            replace_inplace_for_first_innermost_layer_(self.layers)

        if _REQ_GRAD:
            self.input_buffer = {}  # For saving activations
        else:

            def _get_grad(self, micro_batch_idx):
                raise NotImplementedError()

            self.get_grad = types.MethodType(_get_grad, self)

        self.bwd_graph_head_buffer = {}  # For recompute
        self.req_grad = req_grad_dict_to_tuple(req_grad)
        self.outputs_req_grad = req_grad_dict_to_tuple(outputs_req_grad)
        if to_device:
            self.to(self.device)

    def forward(self, x: TensorOrTensors, micro_batch_idx):

        if self.training:
            # EXPLICITLY DO CLONE
            if isinstance(x, Tensor):
                # Note - we clone here because we don't want the tensor to get overridden.
                # TODO: it could be done better if we use multiple input buffers instead of allocating
                # (when #buffers==#max(len(input_buffer)))
                # In pytorch it can happen automatically with THCCashingAlocator.
                # Save activation only if gradient is needed.
                if self._REQ_GRAD:
                    if self._CLONE_INPUTS:
                        x = x.detach().clone().requires_grad_(self.req_grad[0])
                    else:
                        x = x.detach().requires_grad_(self.req_grad[0])

                    self.input_buffer[micro_batch_idx] = x
                x = self.layers(x)
            else:
                if self._REQ_GRAD:
                    if self._CLONE_INPUTS:
                        x = list(get_dcr(x, self.req_grad))
                    else:
                        x = list(get_dr(x, self.req_grad))
                    self.input_buffer[micro_batch_idx] = x

                # UNFLATEN
                x = unflatten(x, self.req_grad)
                x = self.layers(*x)

            # save the head.
            # NOTE: we do not need to unflatten, it happens in filter_for_backward
            self.bwd_graph_head_buffer[micro_batch_idx] = x
            return x

        else:
            with torch.no_grad():
                if isinstance(x, Tensor):
                    x = self.layers(x)
                else:
                    # UNFLATEN
                    x = unflatten(x, self.req_grad)
                    x = self.layers(*x)
                return x

    def recompute(self, micro_batch_idx):
        pass  # HACK: so we don't have to change the code in partition manager

    def backward_from_recomputed(self, g, micro_batch_idx):
        # HACK: misleading name, so we don't have to change the code in partition manager.
        x = self.bwd_graph_head_buffer.pop(micro_batch_idx)
        x, g = filter_for_backward(x, g)
        torch.autograd.backward(x, g)

    def get_grad(self, micro_batch_idx):
        """ returns an iteretable of grads """
        # NOTE: This method can be patched.
        x = self.input_buffer.pop(micro_batch_idx)
        if isinstance(x, Tensor):
            return (x.grad, )
        else:
            return [y.grad for y in filter_req_grad_tensors(x)]


class FirstPartitionWithoutRecomputation(PartitionWithoutRecomputation):
    """ its Just a hack for GPIpe... """
    _CLONE_INPUTS = False  # won't be accesed anyway

    def __init__(self, *args, **kw):
        super().__init__(*args, _REQ_GRAD=False, **kw)


##################
# GPipe
# TODO: we can easly avoid clones() and just use the same buffer.
# NOTE: this already  happens in multiprocessing
# it will save buffer memory, but its minor.
##################


class GPipePartition(nn.Module):
    """ Do not do recomputation on the last micro batch
        we have to know if we are last micro batch at all functions.
     """
    RECOMP_PARTITION_CLS = Partition
    NO_RECOMP_PARTITION_CLS = PartitionWithoutRecomputation
    _CLONE_INPUTS = True

    def __init__(self, *args, **kw):
        super().__init__()
        self.is_last_micro_batch = False

        self.recomputation_partition = self.RECOMP_PARTITION_CLS(*args, **kw)
        self.no_recomputation_partition = self.NO_RECOMP_PARTITION_CLS(
            *args, **kw)
        self.recomputation_partition._CLONE_INPUTS = self._CLONE_INPUTS
        self.no_recomputation_partition._CLONE_INPUTS = self._CLONE_INPUTS

    def forward(self, *args, **kw):
        if self.is_last_micro_batch:
            return self.no_recomputation_partition.forward(*args, **kw)
        else:
            return self.recomputation_partition.forward(*args, **kw)

    def recompute(self, micro_batch_idx):
        if not self.is_last_micro_batch:
            self.recomputation_partition.recompute(micro_batch_idx)

    def backward_from_recomputed(self, g, micro_batch_idx):
        # NOTE: for the last partition this API is not very clear
        # Currently - ,just pass (NONE) as grad_tensor when we do recomputation,
        # or call loss.backward() when we don't.
        if self.is_last_micro_batch:
            self.no_recomputation_partition.backward_from_recomputed(
                g, micro_batch_idx)
        else:
            self.recomputation_partition.backward_from_recomputed(
                g, micro_batch_idx)

    def get_grad(self, micro_batch_idx):
        """ returns an iteretable of grads """
        if self.is_last_micro_batch:
            return self.no_recomputation_partition.get_grad(micro_batch_idx)
        else:
            return self.recomputation_partition.get_grad(micro_batch_idx)

    def pop_saved_graph_head(self, micro_batch_idx):
        """ HACK, TODO: currently, the last partition backprop is done by trainer,
        # as sometimes we have loss_fn and sometimes we don't
        # so the correct behavior is to get the recomputation output (x)
        # and pass it to the trainer.
        # therefore, 
        # self.partition.backward_from_recomputed(None, batch_idx)
        # which does the pop and does loss.backward(), (None is grad_tensor)
        # will not be called
        # so we have to pop ourselves...
        """
        used_partition = self.no_recomputation_partition if self.is_last_micro_batch else self.recomputation_partition
        return used_partition.bwd_graph_head_buffer.pop(micro_batch_idx)


class GPipeFirstPartition(GPipePartition):
    """ Do not do recomputation on the last micro batch """
    RECOMP_PARTITION_CLS = FirstPartition
    NO_RECOMP_PARTITION_CLS = FirstPartitionWithoutRecomputation
    _CLONE_INPUTS = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)


class GPipeLastPartition(GPipePartition):
    """ NOTE: for doing backward_fro_recomputed,just pass (NONE) as grad_tensor """
    RECOMP_PARTITION_CLS = Partition
    NO_RECOMP_PARTITION_CLS = LastPartition
    _CLONE_INPUTS = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, x: TensorOrTensors, micro_batch_idx):
        x = super().forward(x, micro_batch_idx)
        if not isinstance(x, Tensor):
            assert (len(x) == 1)
            return x[0]
        return x


class GPipeLastPartitionWithLabelInput(GPipeLastPartition):
    RECOMP_PARTITION_CLS = Partition
    NO_RECOMP_PARTITION_CLS = LastPartitionWithLabelInput
    _CLONE_INPUTS = True

    # TODO: it is very stupied that we calculate loss for all micro batches in the dummy forward,
    # but its very anoying to fix this.
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)


filter_req_grad_tensors = partial(
    filter, lambda a: isinstance(a, Tensor) and a.requires_grad)


def filter_for_backward(x, g):
    # TODO: remove this compeltly, by saving for backward only whats needed.
    # NOTE: we currently build on this behavior  so be careful when removing.
    x = filter_req_grad_tensors(flatten(x))
    x = list(x)  # FIXME: just for the assert
    g = list(flatten(g))  # FIXME: just for the assert
    assert len(x) == len(g)
    tensors = []
    grad_tensors = []
    for t, gt in zip(x, g):
        if t.grad_fn is not None:
            tensors.append(t)
            grad_tensors.append(gt)
        else:
            if g is not None:
                print("-W- calculated and sent un-needed grad")
    return tensors, grad_tensors
    # torch.autograd.backward(tensors, grad_tensors)


def req_grad_dict_to_tuple(req_grad: dict):
    ret = tuple(v for i, v in req_grad.items())
    # print(f"-I- req_grad tuple: {ret}")
    return ret


# NOTE: we count that the user would not change mutable object because we don't deepcopy them...


def assert_same_size(x, g):
    assert len(list(flatten(x))) == len(list(flatten(g)))


def get_dcr(x, req_grad):
    # assert_same_size(x, req_grad)
    res = []
    for t, r in zip(flatten(x), flatten(req_grad)):
        if isinstance(t, Tensor):
            assert isinstance(r, bool)
            res.append(t.detach().clone().requires_grad_(r))
        else:
            assert r is False
            res.append(t)
    return res


def get_dr(x, req_grad):
    # assert_same_size(x, req_grad)
    res = []
    for t, r in zip(flatten(x), flatten(req_grad)):
        if isinstance(t, Tensor):
            assert isinstance(r, bool)
            res.append(t.detach().requires_grad_(r))
        else:
            assert r is False
            res.append(t)
    return res


def get_r(x, req_grad):
    # assert_same_size(x, req_grad)
    res = []
    for t, r in zip(flatten(x), flatten(req_grad)):
        if isinstance(t, Tensor):
            assert isinstance(r, bool)
            res.append(t.requires_grad_(r))
        else:
            assert r is False
            res.append(t)
    return res
