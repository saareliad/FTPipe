import torch
import torch.nn as nn
import timeit
from torch.autograd import Variable
from collections import namedtuple
from .utils import traverse_model

__all__ = ['profileNetwork']


class Wrapper(nn.Module):
    '''
    a module whose purpose is to profile a given layer,
    measuring forward/backward computation time, and estimated memory consumption

    Parameters
    ----------
    sub_module:
        a nn.module to be profiled

    '''

    def __init__(self, sub_module: nn.Module, device="cuda"):
        super(Wrapper, self).__init__()
        self.layer = sub_module
        self.device = device
        self.forward_time = 0
        self.backward_time = 0
        self.input_size = 0
        self.output_size = 0
        self.param_size, self.buffer_size = self._layer_size()

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
        detached_inputs = map(
            lambda t: Variable(t.data, requires_grad=True).to(self.device).clone(), inputs)

        forward_time, outputs = self._time_op(
            self.layer, *detached_inputs)

        self.forward_time += forward_time

        # reduce outputs to calculate dummy loss
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        for out in outputs:
            loss = loss + out.norm()

        # measure backward execution time
        backward_time, _ = self._time_op(torch.autograd.backward, loss)
        self.backward_time += backward_time

        # input and output size
        self.input_size = 0
        self.output_size = 0
        for t in inputs:
            self.input_size += t.numel()
        for o in outputs:
            self.output_size += o.numel()

        return outputs

    def _time_op(self, func, *inputs):
        exec_time = 0
        if(self.device == "cuda"):
            # milliseconds
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = func(*inputs)
            end.record()
            torch.cuda.synchronize()
            exec_time = (start.elapsed_time(end))
        else:
            # convert to milliseconds
            start = timeit.time.time()
            out = func(*inputs)
            end = timeit.time.time()
            exec_time = 1000*(end - start)

        return exec_time, out


def profileNetwork(net: nn.Module, *sample_batch, basic_block=None, device="cuda", max_depth=100, num_iter=1):
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

    num_iter:
        number of runs the profiler will perform in order to get time measurments

    device:
        the device on which we will profile the network defaults to cuda
    '''
    # wrap all individula layers for profiling
    layers_dict = _wrap_profiled_layers(net, max_depth, basic_block, device)

    # gather all individual layer sizes
    param_sizes = [layer.param_size for layer in layers_dict.values()]
    buffer_sizes = [layer.buffer_size for layer in layers_dict.values()]

    # perform symbolic forward backward run
    for _ in range(num_iter):
        _perform_forward_backward_pass(net, device, *sample_batch)

    # gather forward and backward execution times
    backward_times = [layer.backward_time / num_iter
                      for layer in layers_dict.values()]
    forward_times = [layer.forward_time/num_iter
                     for layer in layers_dict.values()]

    # gather input and output sizes
    layer_input_sizes = [layer.input_size for layer in layers_dict.values()]
    layer_output_sizes = [layer.output_size for layer in layers_dict.values()]

    Profile = namedtuple(
        'Profile', 'forward_time backward_time param_size buffer_size input_size output_size')

    # prepare profiling results
    layers_profile = {name: Profile(forward, backward, param_size, buffer_size, in_size, out_size) for name, forward, backward, param_size, buffer_size, in_size, out_size in zip(
        layers_dict.keys(), forward_times, backward_times, param_sizes, buffer_sizes, layer_input_sizes, layer_output_sizes)}

    _unwrap_layers(net)

    # TODO fix time measurement for first layer
    max_node = list(layers_profile.items())[0]
    for name, node in layers_profile.items():
        if node.forward_time > max_node[1].forward_time:
            max_node = (name, node)
    layers_profile[max_node[0]] = Profile(1, 1, 1, 1, 1, 1)

    return layers_profile


def _perform_forward_backward_pass(net, device, *inputs):
    # move all inputs and outputs to specified device
    out = map(lambda t: t.to(device), inputs)
    net.to(device)
    # warmup
    a = torch.randn(1000, 2000).to(device)
    b = torch.randn(2000, 1000).to(device)
    a.mm(b)
    if device == "cuda":
        torch.cuda.synchronize()

    out = net(*out)

    if device == "cuda":
        torch.cuda.synchronize()

    return out


def _wrap_profiled_layers(module: nn.Module, depth, basic_block, device):
    '''
    wraps all layers specified by depth and basic blocks of module by changing the binding in the network module dictionary
    '''
    layers_dict = {}

    for sub_layer, scope, parent in traverse_model(module, depth, basic_block):
        name = scope[scope.rfind('[')+1:-1]
        parent._modules[name] = Wrapper(sub_layer, device=device)
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
