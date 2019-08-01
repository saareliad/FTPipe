import torch
import torch.nn as nn
import time
from collections import namedtuple
from ..utils import traverse_model, get_device, _detach_inputs, _get_size
from typing import List, Optional, Dict

__all__ = ['profileNetwork', 'Profile']

Profile = namedtuple('Profile',
                     'forward_time backward_time cuda_memory_forward cuda_memory_backward  layer_size')


def profileNetwork(net: nn.Module, *sample_batch, basic_blocks: Optional[List[nn.Module]] = None, max_depth=100) -> Dict[str, Profile]:
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

    basic_blocks:
        a tuple of nn.Module classes that the profiler will regard as a cohesive unit
        for eg. if basic_blocks = nn.Sequential then the profiler will break it down to its components

    max_depth:
        determins how far the profiler will go in the model tree



    '''
    # wrap all individula layers for profiling
    layers_dict = _wrap_profiled_layers(net, max_depth, basic_blocks)

    # perform 2 symbolic forward backward run first one is warmup as we have seen the first time measurements are higher
    _perform_forward_backward_pass(net, *sample_batch)
    _perform_forward_backward_pass(net, *sample_batch)

    # gather forward and backward execution times
    backward_times = [layer.backward_time
                      for layer in layers_dict.values()]
    forward_times = [layer.forward_time
                     for layer in layers_dict.values()]

    # gather input and output sizes
    layer_input_sizes = [layer.input_size for layer in layers_dict.values()]
    layer_output_sizes = [layer.output_size for layer in layers_dict.values()]

    # gather all individual layer sizes
    param_sizes = [layer.parameters_size for layer in layers_dict.values()]
    buffer_sizes = [layer.buffers_size for layer in layers_dict.values()]

    # gather cuda memory consumption
    cuda_memory = [(layer.forward_cuda_mem, layer.backward_cuda_mem)
                   for layer in layers_dict.values()]

    # prepare profiling results
    layers_profile = {name: Profile(forward, backward, *cuda_mem, param_size+buffer_size+in_size+out_size) for name, forward, backward, param_size, buffer_size, in_size, out_size, cuda_mem in zip(
        layers_dict.keys(), forward_times, backward_times, param_sizes, buffer_sizes, layer_input_sizes, layer_output_sizes, cuda_memory)}

    _unwrap_layers(net)

    return layers_profile


def _perform_forward_backward_pass(net, *sample_batch):
    device = get_device(sample_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
        out = net(*sample_batch)
        torch.cuda.synchronize(device=device)
    else:
        out = net(*sample_batch)

    return out


def _wrap_profiled_layers(module: nn.Module, depth, basic_blocks: List[nn.Module]):
    layers_dict = {}

    for sub_layer, scope, parent in traverse_model(module, depth, basic_blocks):
        name = scope[scope.rfind('[')+1:-1]
        parent._modules[name] = Wrapper(sub_layer)
        layers_dict[scope] = parent._modules[name]

    return layers_dict


def _unwrap_layers(module: nn.Module):
    for name, sub_module in module._modules.items():
        if isinstance(sub_module, Wrapper):
            module._modules[name] = sub_module.layer
        else:
            _unwrap_layers(sub_module)


class Wrapper(nn.Module):
    '''
    a module whose purpose is to profile a given layer\n
    when the wrapper performs forward propagation it records the following metrics:\n
        forward_time: the execution time of a forward pass of the underlying layer in milliseconds\n
        backward_time: the execution time of a backward pass of the underlying layer in milliseconds\n
        input_size: the input size in GB
        output_size: the layer output size in GB
        parameters_size: the size of parameters of the layer in GB
        buffers_size: the size of buffers of the layer in GB
        forward_cuda_mem: the peak CUDA memory usage during the forward pass in GB
        backward_cuda_mem: the peak CUDA memory usage during the backward pass in GB

    Parameters
    ----------
    sub_module:
        a nn.module to be profiled

    '''

    def __init__(self, sub_module: nn.Module):
        super(Wrapper, self).__init__()
        self.layer = sub_module
        self.forward_time = 0
        self.backward_time = 0
        self.input_size = 0
        self.output_size = 0
        self.parameters_size, self.buffers_size = self._layer_size()
        self.forward_cuda_mem = 0
        self.backward_cuda_mem = 0

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

    def forward(self, *inputs):
        '''
        perform forward and backward pass of the underlying layer and measure metrics
        '''

        # detach inputs from previous history enabling us to measure execution time
        # only for this layer
        device = get_device(inputs)
        # detached_inputs = map(lambda t: t.detach(), inputs)
        detached_inputs = _detach_inputs(inputs)

        self.forward_time, outputs, self.forward_cuda_mem = self._time_op(
            self.layer, *detached_inputs)

        # reduce outputs to calculate dummy loss
        loss = torch.zeros(1, requires_grad=True, device=device)
        for out in outputs:
            loss = loss + out.norm()

        # measure backward execution time
        self.backward_time, _,  self.backward_cuda_mem = self._time_op(
            torch.autograd.backward, loss)

        # input and output size
        self.input_size = _get_size(inputs)
        self.output_size = _get_size(outputs)

        #size in Gigabaytes
        self.backward_cuda_mem /= 1e9
        self.forward_cuda_mem /= 1e9
        self.input_size /= 1e9
        self.output_size /= 1e9
        self.parameters_size /= 1e9
        self.buffers_size /= 1e9

        return outputs

    def _time_op(self, func, *inputs):
        exec_time = 0
        cuda_mem = 0
        device = get_device(inputs)
        if(device.type == 'cuda'):
            # milliseconds
            torch.cuda.reset_max_memory_allocated(device=device)
            torch.cuda.synchronize(device=device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = func(*inputs)
            end.record()
            torch.cuda.synchronize(device=device)
            exec_time = (start.elapsed_time(end))
            cuda_mem = torch.cuda.max_memory_allocated(device=device)
        else:
            # convert seconds to milliseconds
            start = time.time()
            out = func(*inputs)
            end = time.time()
            exec_time = 1000*(end - start)

        return exec_time, out, cuda_mem
