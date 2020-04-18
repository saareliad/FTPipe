import time
from collections import namedtuple
from typing import Dict, List, Optional
from itertools import chain
import torch
import torch.nn as nn

from ..utils import Tensors, _detach_inputs, _get_size_and_shape, get_device, traverse_model, flatten, _get_dtypes

__all__ = ['profile_network', 'Profile']

Profile = namedtuple(
    'Profile',
    'forward_time backward_time cuda_memory_forward cuda_memory_backward layer_size input_size output_size input_shape output_shape input_dtype output_dtype'
)


def profile_network(
        net: nn.Module,
        sample_batch: Tensors,
        kwargs: Optional[Dict] = None,
        basic_blocks: Optional[List[nn.Module]] = None,
        max_depth=100,
        n_iter=10,
        save_memory_mode=False,
        recomputation=False,
        force_no_recomp_scopes=lambda s: False,
) -> Dict[str, Profile]:
    '''
    profiles a network's computation time(forward/backward) and memory consumption
    returns a dictionary from layer_scope to Profile

    Parameters
    ----------
    net:
        the network we wish to profile a nn.Module

    sample_batch:
        a sample batch that will be used to measure executation time of network
        can be single/multiple inputs

    kwargs:
        keyword args to pass to the profiled model

    basic_blocks:
        a tuple of nn.Module classes that the profiler will regard as a cohesive unit
        for eg. if basic_blocks = nn.Sequential then the profiler will break it down to its components

    max_depth:
        determins how far the profiler will go in the model tree

    n_iter:
        number of iteration to use for profiling
        the profiling will be averaged accross all iterations

    '''
    if kwargs is None:
        kwargs = {}
    if not isinstance(sample_batch, tuple):
        sample_batch = (sample_batch, )

    # wrap all individula layers for profiling
    layers_dict = _wrap_profiled_layers(net,
                                        max_depth,
                                        basic_blocks,
                                        save_memory_mode=save_memory_mode,
                                        recomputation=recomputation,
                                        force_no_recomp_scopes=force_no_recomp_scopes)

    # perform n_iter symbolic forward backward run
    # first one is warmup as we have seen the first time measurements are higher
    for _ in range(n_iter + 1):
        _perform_forward_backward_pass(net, *sample_batch, **kwargs)

    # gather forward and backward execution times
    backward_times = [
        layer.avg_time(forward=False) for layer in layers_dict.values()
    ]
    forward_times = [
        layer.avg_time(forward=True) for layer in layers_dict.values()
    ]

    # gather input and output sizes
    layer_input_sizes = [layer.input_size for layer in layers_dict.values()]
    layer_output_sizes = [layer.output_size for layer in layers_dict.values()]

    # gather all individual layer sizes
    param_sizes = [layer.parameters_size for layer in layers_dict.values()]
    buffer_sizes = [layer.buffers_size for layer in layers_dict.values()]

    # gather cuda memory consumption
    cuda_memory = [(layer.forward_cuda_mem, layer.backward_cuda_mem)
                   for layer in layers_dict.values()]

    # gather input/output shapes
    input_shapes = [layer.input_shape for layer in layers_dict.values()]
    output_shapes = [layer.output_shape for layer in layers_dict.values()]

    # gather input/output dtypes
    input_dtypes = [layer.input_dtype for layer in layers_dict.values()]
    output_dtypes = [layer.output_dtype for layer in layers_dict.values()]

    # prepare profiling results
    layers_profile = {
        name: Profile(forward, backward, *cuda_mem, param_size + buffer_size,
                      in_size, out_size, in_shape, out_shape, input_dtype, output_dtype)
        for name, forward, backward, param_size, buffer_size,
        in_size, out_size, cuda_mem, in_shape, out_shape, input_dtype, output_dtype in zip(
            layers_dict.keys(), forward_times, backward_times, param_sizes,
            buffer_sizes, layer_input_sizes, layer_output_sizes, cuda_memory,
            input_shapes, output_shapes, input_dtypes, output_dtypes)
    }

    _unwrap_layers(net)

    return layers_profile


def _perform_forward_backward_pass(net, *sample_batch: Tensors,
                                   **kwargs: Dict):
    if len(sample_batch) > 0:
        device = get_device(sample_batch)
    else:
        for t in kwargs.values():
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
        out = net(*sample_batch, **kwargs)
        torch.cuda.synchronize(device=device)
    else:
        out = net(*sample_batch, **kwargs)
    # delete gradients
    for p in net.parameters():
        assert p.grad is None

    return out


def _wrap_profiled_layers(module: nn.Module,
                          depth,
                          basic_blocks: List[nn.Module],
                          save_memory_mode=False,
                          recomputation=False,
                          force_no_recomp_scopes=lambda s: False
                          ):
    layers_dict = {}
    for sub_layer, scope, parent in traverse_model(module,
                                                   depth,
                                                   basic_blocks=basic_blocks):
        name = scope[scope.rfind('[') + 1:-1]

        scope_specific_recomp = recomputation
        if force_no_recomp_scopes(scope):
            scope_specific_recomp = False

        wrapper = Wrapper(sub_layer,
                          scope,
                          save_memory_mode=save_memory_mode,
                          recomputation=scope_specific_recomp)
        parent.add_module(name, wrapper)
        layers_dict[scope] = wrapper

    return layers_dict


def _unwrap_layers(module: nn.Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, Wrapper):
            sub_module.on_unwrap()
            module.add_module(name, sub_module.layer)
        else:
            _unwrap_layers(sub_module)


class Wrapper(nn.Module):
    '''
    a module whose purpose is to profile a given layer\n
    when the wrapper performs forward propagation it records the following metrics:\n
        forward_time: the execution time of a forward pass of the underlying layer in milliseconds\n
        backward_time: the execution time of a backward pass of the underlying layer in milliseconds\n
        input_size: the input size in MB
        output_size: the layer output size in MB
        parameters_size: the size of parameters of the layer in MB
        buffers_size: the size of buffers of the layer in MB
        forward_cuda_mem: the peak CUDA memory usage during the forward pass in MB
        backward_cuda_mem: the peak CUDA memory usage during the backward pass in MB

    Parameters
    ----------
    sub_module:
        a nn.module to be profiled

    '''

    def __init__(self,
                 sub_module: nn.Module,
                 scope: str,
                 save_memory_mode=False,
                 recomputation=False):
        super(Wrapper, self).__init__()
        assert isinstance(recomputation, bool)
        self.layer = sub_module
        self.forward_time = []
        self.backward_time = []
        self.input_size = 0
        self.output_size = 0
        self.parameters_size, self.buffers_size = self._layer_size()
        self.forward_cuda_mem = 0
        self.backward_cuda_mem = 0
        self.input_shape = []
        self.output_shape = []
        self.input_dtype = []
        self.output_dtype = []
        self.scope = scope
        self.save_memory_mode = save_memory_mode
        self.recomputation = recomputation  # TODO

        if save_memory_mode:
            self.layer.to('cpu')

    def _layer_size(self):
        '''
        return the size of the layer considering parameters and buffers
        '''
        parameters_size = buffers_size = 0
        for param in self.layer.parameters():
            parameters_size += param.nelement() * param.element_size()
        for buffer in self.layer.buffers():
            buffers_size += buffer.nelement() * buffer.element_size()

        return parameters_size, buffers_size

    def forward(self, *inputs: Tensors, **kwargs: Dict):
        '''
        perform forward and backward pass of the underlying layer and measure metrics
        '''
        ts = list(chain(self.parameters(), self.buffers()))
        if len(ts) > 0 and not self.save_memory_mode:
            device = ts[0].device
        elif len(inputs) > 0:
            device = get_device(inputs)
        else:
            for t in kwargs.values():
                if isinstance(t, torch.Tensor):
                    device = t.device
                    break
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = torch.device(device)

        if self.save_memory_mode:
            self.layer.to(self.device)

        # detach inputs from previous history enabling us to measure execution time
        # only for this layer

        detached_inputs = _detach_inputs(inputs)
        # TODO: we  the input as requires grad, as this is mostly the case,
        #  the grad has to be passed backward.
        # However, then gradient creation will be computed for all

        with torch.set_grad_enabled(not self.recomputation):
            # if recomputation: its a dummy forward
            forward_time, outputs, self.forward_cuda_mem = time_op(self.device,
                                                                   self.layer, *detached_inputs, **kwargs)

        self.forward_time.append(forward_time)


        if self.recomputation:
            # Then, we do fwd+bwd
            # FIXME: self.forward_cuda_mem...
            forward_time, outputs, self.forward_cuda_mem = time_op(self.device,
                                                                   self.layer, *detached_inputs, **kwargs)
        
        # NOTE: the commented code is less accurate, but it can be usefull for memory problems
        # reduce outputs to calculate dummy loss
        # loss = torch.zeros(1, requires_grad=False, device=device)
        # for out in flatten(outputs):
        #     if isinstance(out, torch.Tensor):
        #         loss = loss + out.sum()
        # Loss makes sense only for the last layer...

        # calculate dummy loss
        flattened_outputs = flatten(outputs)
        grad_tensors = []
        has_grad_fn = False
        for out in flatten(outputs):
            if isinstance(out, torch.Tensor):
                grad_tensors.append(torch.randn_like(out))
                if (out.grad_fn is not None) or out.requires_grad:
                    has_grad_fn = True
            else:
                grad_tensors.append(None)
        
        # measure backward execution time

        # if loss.grad_fn is not None or loss.requires_grad:
        if has_grad_fn:
            backward_time, _, self.backward_cuda_mem = time_op(self.device,
                                                               torch.autograd.backward,
                                                               tensors=flattened_outputs,
                                                               grad_tensors=grad_tensors)

            # TODO: also create option to check gradient accumulation,
            #  in case this is the domminant case

            # delete gradients to save memory after backward.
            for p in self.parameters():
                p.grad = None
            # TODO: can also clean inputs grad
            for p in detached_inputs:
                p.grad = None
        else:
            backward_time, self.backward_cuda_mem = 0.0, 0.0

        if self.recomputation:
            backward_time = forward_time + backward_time

        self.backward_time.append(backward_time)

        # input and output size
        self.input_size, self.input_shape = _get_size_and_shape(inputs)
        self.output_size, self.output_shape = _get_size_and_shape(outputs)

        self.input_dtype = _get_dtypes(inputs)
        self.output_dtype = _get_dtypes(outputs)

        if isinstance(self.input_shape, torch.Size):
            self.input_shape = (self.input_shape, )
        if isinstance(self.output_shape, torch.Size):
            self.output_shape = (self.output_shape, )

        if isinstance(self.input_dtype, torch.dtype):
            self.input_dtype = (self.input_dtype, )
        if isinstance(self.output_dtype, torch.dtype):
            self.output_dtype = (self.output_dtype, )

        # size in MegaBytes
        self.input_size /= 1e6
        self.output_size /= 1e6
        self.parameters_size /= 1e6
        self.buffers_size /= 1e6

        if self.save_memory_mode:
            self.layer.to('cpu')

        return outputs

    def avg_time(self, forward=False):
        if forward:
            return avg_time(self.forward_time)
        else:
            return avg_time(self.backward_time)

    # just in case those operations are required we pass them to the profiled layer

    def __iter__(self):
        return iter(self.layer)

    def __getitem__(self, key):
        return self.layer[key]

    def __setitem__(self, key, value):
        self.layer[key] = value

    def __delitem__(self, idx):
        delattr(self.layer, idx)

    def __len__(self):
        return len(self.layer)

    def __contains__(self, key):
        return key in self.layer

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except Exception:
            return getattr(self.layer, name)

    def on_unwrap(self):
        if self.save_memory_mode:
            self.layer.to('cuda')  # HACK, assuming its called only at cuda.


def time_op(device, func, *inputs: Tensors, **kwargs: Dict):
    exec_time = 0
    cuda_mem = 0
    if (device.type == 'cuda'):
        torch.cuda.reset_max_memory_allocated(device=device)
        base_mem = torch.cuda.max_memory_allocated(device=device)

        # measure execution time
        torch.cuda.synchronize(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = func(*inputs, **kwargs)
        end.record()
        torch.cuda.synchronize(device=device)
        exec_time = (start.elapsed_time(end))

        # record memory usage
        peak_usage = torch.cuda.max_memory_allocated(device=device)
        cuda_mem = peak_usage - base_mem
    else:
        # convert seconds to milliseconds
        start = time.time()
        out = func(*inputs, **kwargs)
        end = time.time()
        exec_time = 1000 * (end - start)

    return exec_time, out, cuda_mem


def avg_time(times):
    max_v = max(times)

    return sum([t for t in times if t < max_v
                ]) / (len(times) - 1)
