import torch
import torch.nn as nn
import timeit
# import metis


class Wrapper(nn.Module):
    '''
    a module whose purpose is to profile a given layer,
    measuring forward/backward computation time, and estimated memory consumption

    Parameters
    ----------
    sub_module:
        a nn.module to be profiled

    '''

    def __init__(self, sub_module: nn.Module, idx=0, device="cuda"):
        super(Wrapper, self).__init__()
        self.layer = sub_module
        self.device = device
        self.idx = idx
        self.forward_time = None
        self.backward_timestamp = None

        def backward_time_hook(_, __, ___):
            # record when the layer finished calculating gradients
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.backward_timestamp = timeit.time.time()

        self.layer.register_backward_hook(backward_time_hook)

        self.size = self._layer_size()

    def _layer_size(self):
        '''
        return the size of the layer considering parameters and buffers
        '''
        # TODO maybe include activations/gradients size
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
        forward_time = 0
        out = None
        if(self.device == "cuda"):
            # milliseconds
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self.layer(*inputs)
            end.record()
            torch.cuda.synchronize()
            forward_time = (start.elapsed_time(end))
        else:
            # convert to milliseconds
            start = timeit.time.time()
            out = self.layer(*inputs)
            end = timeit.time.time()
            forward_time = 1000*(end - start)

        self.forward_time = forward_time

        return out


class NetProfiler(nn.Module):
    '''
    a module who profiles a network's computation time(forward/backward) and memory consumption
    done via wrapping all layers of the network with a special Wrapper module

    Parameters
    ----------
    net:
        the network we wish to profile a nn.Module

    sample_batch:
        a sample batch that will be used to measure executation time of network
        can be single/multiple inputs

    basic_block:
        a tuple of nn.Module classes that the profiler will regard as a cohesive unit
        for eg. if basic_block = nn.Sequential then the profiler will break it down to it's components

    device:
        the device on which we will profile the network defaults to cuda

    '''
    # TODO maybe include activations/gradients size
    # TODO find better way to measure the backward computation time
    # TODO maybe measure forward+backward together via usage of detach in order to isolate different layers
    # TODO check if it is realy neccessary to call cuda.synchronize all the time

    def __init__(self, net: nn.Module, *sample_batch, basic_block=None, device="cuda", max_depth=None):
        super(NetProfiler, self).__init__()
        self.network = net
        self.basic_block = basic_block
        self.device = device
        self.max_depth = max_depth if max_depth != None else 100
        self._profile(*sample_batch)

    def forward(self, *inputs):
        # move all inputs and outputs to specified device
        out = map(lambda t: t.to(self.device), inputs)
        self.to(self.device)
        # warmup
        a = torch.randn(1000, 2000).to(self.device)
        b = torch.randn(2000, 1000).to(self.device)
        a.mm(b)
        if self.device == "cuda":
            torch.cuda.synchronize()

        out = self.network(*out)

        if self.device == "cuda":
            torch.cuda.synchronize()

        return out

    def _wrap_individual_layers(self, module: nn.Module, depth, idx, layers_dict):
        '''
        wraps all layers of module by changing the binding in the network module dictionary
        '''

        for name, sub_module in module._modules.items():
            if len(list(sub_module.children())) == 0 or (self.basic_block != None and isinstance(sub_module, self.basic_block)) or depth == 0:
                module._modules[name] = Wrapper(sub_module, idx, self.device)
                layers_dict[idx] = module._modules[name]
                idx += 1
            else:
                idx, layers_dict = self._wrap_individual_layers(
                    sub_module, depth-1, idx, layers_dict)
        return idx, layers_dict

    def _unwrap_layers(self, module: nn.Module, idx):
        '''
        return all binding to what they were originally, removes all Wrappers from self.network
        '''
        for name, sub_module in module._modules.items():
            if isinstance(sub_module, Wrapper):
                module._modules[name] = sub_module.layer
                self.layers[idx] = module._modules[name]
                idx += 1
            else:
                idx = self._unwrap_layers(
                    sub_module, idx)
        return idx

    def _profile(self, *sample_batch):
        '''
        profiles the network using a sample input
        '''
        # wrap all individula layers for profiling
        self.num_layers, self.layers = self._wrap_individual_layers(
            self.network, self.max_depth, 0, {})

        # gather all individual layer sizes
        self.layer_sizes = [layer.size for layer in self.layers.values()]

        # perform symbolic forward run
        outputs = self(*sample_batch)

        # gather forward times results
        self._forward_times = [
            layer.forward_time for layer in self.layers.values()]

        # perform symbolic backward run
        loss = 0.0
        for out in outputs:
            loss += out.norm()

        if self.device == "cuda":
            torch.cuda.synchronize()

        loss.backward()

        if self.device == "cuda":
            torch.cuda.synchronize()

        # gather backward times results
        # time is calculated by subtracting successive timestamps
        # for lack of a better option the last layer performs backpropagation in 0 time
        self.backward_times = [
            layer.backward_timestamp for layer in self.layers.values()]

        self.backward_times = [
            end-start for end, start in zip(self.backward_times, self.backward_times[1:])]+[0]
