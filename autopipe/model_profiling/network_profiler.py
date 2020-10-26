import time
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from ..utils import flatten, nested_map, traverse_model, get_device, ExecTimes

__all__ = ['profile_network']

# TODO: DEPRECATED
def profile_network(
        net: nn.Module,
        sample_batch: tuple = (),
        kwargs: Optional[Dict] = None,
        basic_blocks: Optional[List[nn.Module]] = None,
        max_depth=100,
        n_iter=10,
        save_memory_mode=False,
        recomputation=False,
        force_no_recomp_scopes=None,
) -> Dict[str, ExecTimes]:
    """
    profiles a network's computation time(forward/backward)
    returns a dictionary from layer_scope to ExecTimes

    Parameters
    ----------
    net:
        the network we wish to profile a nn.Module

    sample_batch:
        a sample batch that will be used to measure execution time of network
        can be single/multiple inputs

    kwargs:
        keyword args to pass to the profiled model

    basic_blocks:
        a tuple of nn.Module classes that the profiler will regard as a cohesive unit
        for eg. if basic_blocks = nn.Sequential then the profiler will break it down to its components

    max_depth:
        determines how far the profiler will go in the model tree

    n_iter:
        number of iteration to use for profiling
        the profiling will be averaged across all iterations, after throwing several outliers

    """
    if kwargs is None:
        kwargs = {}
    if basic_blocks is None:
        basic_blocks = ()
    if not isinstance(sample_batch, tuple):
        sample_batch = (sample_batch,)

    if force_no_recomp_scopes is None:

        def f(s):
            return False
    else:
        f = force_no_recomp_scopes

    # wrap all individual layers for profiling
    layers_dict = _wrap_profiled_layers(net,
                                        max_depth,
                                        basic_blocks,
                                        save_memory_mode=save_memory_mode,
                                        recomputation=recomputation,
                                        force_no_recomp_scopes=f)

    # perform n_iter symbolic forward backward run
    # first one is warmup as we have seen the first time measurements are higher
    for _ in range(n_iter + 1):
        _perform_forward_backward_pass(net,
                                       *sample_batch,
                                       save_memory_mode=save_memory_mode,
                                       **kwargs)

    # gather forward and backward execution times
    backward_times = [
        layer.avg_time(forward=False) for layer in layers_dict.values()
    ]
    forward_times = [
        layer.avg_time(forward=True) for layer in layers_dict.values()
    ]

    # prepare profiling results
    layers_profile = {
        name: ExecTimes(forward, backward)
        for name, forward, backward in zip(layers_dict.keys(), forward_times,
                                           backward_times)
    }

    _unwrap_layers(net)

    return layers_profile


def _perform_forward_backward_pass(net,
                                   *sample_batch: tuple,
                                   save_memory_mode=False,
                                   **kwargs: Dict):
    if save_memory_mode:
        device = torch.device("cuda")
    else:
        device = get_device((sample_batch, kwargs))

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
                          force_no_recomp_scopes=lambda s: False):
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
    """
    A module whose purpose is to profile a given layer
    when the wrapper performs forward propagation it records the following metrics:
        forward_time: the execution time of a forward pass of the underlying layer in milliseconds
        backward_time: the execution time of a backward pass of the underlying layer in milliseconds
    with slight changes when recomputation is set to True.
    """

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
        self.scope = scope
        self.save_memory_mode = save_memory_mode
        self.recomputation = recomputation
        self.device = None

        if save_memory_mode:
            self.layer.to('cpu')

    def forward(self, *inputs: tuple, **kwargs: Dict):
        """
        Perform forward and backward pass of the underlying layer and measure metrics
        """

        if self.save_memory_mode:
            self.device = torch.device("cuda")
        else:
            self.device = get_device(
                (inputs, kwargs, self.parameters(), self.buffers()))

        if self.save_memory_mode:
            self.layer.to(self.device)

        # detach inputs from previous history enabling us to measure execution time
        # only for this layer
        # Tensor inputs are already detach, here comes special detach for torch.nn.Parameters
        detached_inputs = set_req_grad_for_parameters(inputs)
        # TODO: for shared weights we count the gradient creation twice, even though it could be accumulation 2nd time.
        with torch.set_grad_enabled(not self.recomputation):
            # if recomputation: its a dummy forward
            forward_time, outputs, _ = time_op(self.device, self.layer,
                                               *detached_inputs, **kwargs)

        self.forward_time.append(forward_time)

        if self.recomputation:
            # Then, we do fwd+bwd
            forward_time, outputs, _ = time_op(self.device, self.layer,
                                               *detached_inputs, **kwargs)

        # NOTE: the commented code is less accurate, but it can be useful for memory problems
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
            backward_time, _, _ = time_op(self.device,
                                          torch.autograd.backward,
                                          tensors=flattened_outputs,
                                          grad_tensors=grad_tensors)

            # TODO: also create option to check gradient accumulation,
            #  in case this is the dominant case

            # delete gradients to save memory after backward.
            for p in self.parameters():
                p.grad = None
            # TODO: can also clean inputs grad
            for p in detached_inputs:
                p.grad = None
        else:
            backward_time, _ = 0.0, 0.0

        if self.recomputation:
            backward_time = forward_time + backward_time

        self.backward_time.append(backward_time)

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


def time_op(device, func, *inputs: tuple, **kwargs):
    cuda_mem = 0
    if device.type == 'cuda':
        torch.cuda.reset_max_memory_allocated(device=device)
        base_mem = torch.cuda.max_memory_allocated(device=device)

        # measure execution time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device=device)
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

    return sum([t for t in times if t < max_v]) / (len(times) - 1)


def set_req_grad_for_parameters(ts):
    """ For model parameters which are sent across the pipeline, grad requirements at profiling are always true
        # TODO: support freezing
    """
    def f(t):
        if not isinstance(t, torch.Tensor):
            return t
        req_grad = t.requires_grad if isinstance(t,
                                                 torch.nn.Parameter) else False
        return t.detach().requires_grad_(req_grad)

    return nested_map(f, ts)
