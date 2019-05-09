import torch
import torch.nn as nn
import timeit
from torch.autograd import Variable
from collections import namedtuple
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
        self.size = self._layer_size()
        # TODO normalize weights

    def _layer_size(self):
        '''
        return the size of the layer considering parameters and buffers
        '''
        size = 0
        for param in self.layer.parameters():
            size += param.nelement() * param.element_size()
        for buffer in self.layer.buffers():
            size += buffer.nelement() * buffer.element_size()

        return size

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

    # TODO maybe include activations/gradients size


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
        for eg. if basic_block = nn.Sequential then the profiler will break it down to it's components

    num_iter:
        number of runs the profiler will perform in order to get time measurments

    device:
        the device on which we will profile the network defaults to cuda

    '''
    # wrap all individula layers for profiling
    model_class_name = type(net).__name__
    layers_dict = _wrap_profiled_layers(
        net, max_depth, model_class_name, basic_block, device)

    # gather all individual layer sizes
    layer_sizes = [layer.size for layer in layers_dict.values()]

    # perform symbolic forward backward run
    for _ in range(num_iter):
        _perform_forward_backward_pass(net, device, *sample_batch)

    # gather forward and backward execution times
    backward_times = [layer.backward_time / num_iter
                      for layer in layers_dict.values()]
    forward_times = [layer.forward_time/num_iter
                     for layer in layers_dict.values()]

    Profile = namedtuple('Profile', 'forward_time backward_time size')

    # prepare profiling results
    layers_profile = {name: Profile(forward, backward, size) for name, forward, backward, size in zip(
        layers_dict.keys(), forward_times, backward_times, layer_sizes)}

    _unwrap_layers(net)
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


def _wrap_profiled_layers(module: nn.Module, depth, prefix, basic_block, device):
    '''
    wraps all layers specified by depth and basic blocks of module by changing the binding in the network module dictionary
    '''
    layers_dict = {}
    for name, sub_module in module._modules.items():
        # assume no cyclic routes in the network
        # a module with no children is a layer
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            sub_module_name = f"{prefix}/{type(sub_module).__name__}[{name}]"
            module._modules[name] = Wrapper(sub_module, device=device)
            layers_dict[sub_module_name] = module._modules[name]
        else:
            layers_dict.update(_wrap_profiled_layers(sub_module, depth-1, prefix +
                                                     "/"+type(sub_module).__name__+f"[{name}]", basic_block, device))
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
