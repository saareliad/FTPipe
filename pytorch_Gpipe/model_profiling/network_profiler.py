import torch
import torch.nn as nn
import timeit
from collections import namedtuple
from ..utils import traverse_model
from typing import List, Optional, Dict

__all__ = ['profileNetwork']

Profile = namedtuple(
    'Profile', 'forward_time backward_time cuda_memory layer_size')


def profileNetwork(net: nn.Module, *sample_batch, basic_block: Optional[List[nn.Module]] = None, max_depth=100) -> Dict[str, Profile]:
    '''
    profiles a network's computation time(forward/backward) and memory consumption
    done via wrapping all layers of the network with a special Wrapper module

    Parameters
    ----------
    net:
        the network we wish to profile a nn.Module

    sample_batch:
        a sample batch that will be used to measure executation time of network
        can be single/multiple inputs

    max_depth:
        determins how far the profiler will go in the model tree

    basic_block:
        a tuple of nn.Module classes that the profiler will regard as a cohesive unit
        for eg. if basic_block = nn.Sequential then the profiler will break it down to its components

    '''
    # wrap all individula layers for profiling
    layers_dict = _wrap_profiled_layers(net, max_depth, basic_block)

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
    param_sizes = [layer.param_size for layer in layers_dict.values()]
    buffer_sizes = [layer.buffer_size for layer in layers_dict.values()]

    # gather cuda memory consumption
    cuda_memory = [(layer.forward_cuda_mem, layer.backward_cuda_mem)
                   for layer in layers_dict.values()]

    # prepare profiling results
    layers_profile = {name: Profile(forward, backward, cuda_mem, param_size+buffer_size+in_size+out_size) for name, forward, backward, param_size, buffer_size, in_size, out_size, cuda_mem in zip(
        layers_dict.keys(), forward_times, backward_times, param_sizes, buffer_sizes, layer_input_sizes, layer_output_sizes, cuda_memory)}

    _unwrap_layers(net)

    return layers_profile


def _perform_forward_backward_pass(net, *inputs):
    # move all inputs and outputs to specified device
    device = inputs[0].device
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    out = net(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    return out


def _wrap_profiled_layers(module: nn.Module, depth, basic_block: List[nn.Module]):
    '''
    wraps all layers specified by depth and basic blocks of module by changing the binding in the network module dictionary
    '''
    layers_dict = {}

    for sub_layer, scope, parent in traverse_model(module, depth, basic_block):
        name = scope[scope.rfind('[')+1:-1]
        parent._modules[name] = Wrapper(sub_layer)
        layers_dict[scope] = parent._modules[name]

    return layers_dict


def _unwrap_layers(module: nn.Module):
    '''
    return all binding to what they were originally, removes all Wrappers from self.network
    '''
    for name, sub_module in module._modules.items():
        # assume no cyclic routes in the network
        # a module with no children is a layer
        if isinstance(sub_module, Wrapper):
            module._modules[name] = sub_module.layer
        else:
            _unwrap_layers(sub_module)


class Wrapper(nn.Module):
    '''
    a module whose purpose is to profile a given layer,
    measuring forward/backward computation time, and estimated memory consumption

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
        self.param_size, self.buffer_size = self._layer_size()
        self.forward_cuda_mem = 0
        self.backward_cuda_mem = 0

    def _layer_size(self):
        '''
        return the size of the layer considering parameters and buffers
        '''
        param_size = buffer_size = 0
        for param in self.layer.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.layer.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return param_size, buffer_size

    def forward(self, *inputs):
        '''
        measures the time in milliseconds it took for the module to complete forward computation
        '''
        # detach inputs from previous history enabling us to measure execution time
        # only for this layer
        device = inputs[0].device
        detached_inputs = map(lambda t: t.detach(), inputs)

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
        self.input_size = 0
        self.output_size = 0
        for t in inputs:
            self.input_size += t.nelement() * t.element_size()
        for o in outputs:
            self.output_size += o.nelement() * o.element_size()

        #size in Gigabaytes
        self.backward_cuda_mem /= 1e9
        self.forward_cuda_mem /= 1e9
        self.input_size /= 1e9
        self.output_size /= 1e9
        self.param_size /= 1e9
        self.buffer_size /= 1e9

        return outputs

    def _time_op(self, func, *inputs):
        exec_time = 0
        cuda_mem = 0
        device = inputs[0].device
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
            # convert to milliseconds
            start = timeit.time.time()
            out = func(*inputs)
            end = timeit.time.time()
            exec_time = 1000*(end - start)

        return exec_time, out, cuda_mem
