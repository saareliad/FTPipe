from collections import defaultdict
from itertools import chain
import torch
from torch import Tensor
from .tracer import NodeTypes
from ..utils import nested_map


class LayerProfiler():
    def __init__(self, recomputation=False, n_iter=10, force_no_recomp_scopes=None):
        self.forward_times = defaultdict(list)
        self.backward_times = defaultdict(list)
        self.recomputation = recomputation
        assert n_iter > 0
        self.n_iter = n_iter + 1

        if force_no_recomp_scopes is None:
            self.force_no_recomp_scopes = lambda s: False
        else:
            self.force_no_recomp_scopes = force_no_recomp_scopes

    def time_forward(self, node, function, args, kwargs):
        if LayerProfiler.should_profile(node, function, args, kwargs):
            recomputation = self.recomputation and (
                not self.force_no_recomp_scopes(node.scope))
            for _ in range(self.n_iter):
                args, kwargs = LayerProfiler.detach_tensors((args, kwargs))
                with torch.set_grad_enabled(not recomputation):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(device='cuda')

                    start.record()
                    function(*args, **kwargs)
                    end.record()

                    torch.cuda.synchronize(device='cuda')
                    self.forward_times[node].append(start.elapsed_time(end))

        return None, None

    def time_backward(self, node, function, args, kwargs, output):
        if LayerProfiler.should_profile(node, function, args, kwargs, output=output):
            recomputation = not self.force_no_recomp_scopes(node.scope)
            recomputation = recomputation and self.recomputation

            if not recomputation:
                self.backward_no_recomputation(node, function,
                                               args, kwargs,
                                               output)
            else:
                self.backward_recomputation(node, function,
                                            args, kwargs,
                                            output)

        return output

    def backward_no_recomputation(self, node, function, args, kwargs, output):
        for _ in range(self.n_iter):
            torch.cuda.synchronize(device='cuda')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            tensors = self.only_tensors_that_require_grad(output)
            grads = LayerProfiler.get_grads(tensors)

            torch.cuda.synchronize(device='cuda')
            start.record()
            torch.autograd.backward(tensors=tensors,
                                    grad_tensors=grads,
                                    retain_graph=True)
            end.record()
            torch.cuda.synchronize(device='cuda')

            self.backward_times[node].append(start.elapsed_time(end))

            if node.type is NodeTypes.LAYER:
                for p in function.parameters():
                    p.grad = None
            for p in LayerProfiler.only_tensors_that_require_grad((args, kwargs)):
                p.grad = None

    def backward_recomputation(self, node, function, args, kwargs, output):
        for _ in range(self.n_iter):
            args, kwargs = LayerProfiler.detach_tensors((args, kwargs))
            with torch.enable_grad():
                torch.cuda.synchronize(device='cuda')
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device='cuda')

                start.record()
                output = function(*args, **kwargs)
                tensors = self.only_tensors_that_require_grad(output)
                grads = LayerProfiler.get_grads(tensors)
                torch.autograd.backward(tensors=tensors,
                                        grad_tensors=grads)
                end.record()

                torch.cuda.synchronize(device='cuda')
                self.backward_times[node].append(start.elapsed_time(end))

                if node.type is NodeTypes.LAYER:
                    for p in function.parameters():
                        p.grad = None
                for p in LayerProfiler.only_tensors_that_require_grad((args, kwargs)):
                    p.grad = None

    def print_times(self, backward=False):
        if backward:
            ts = self.backward_times
        else:
            ts = self.forward_times
        for n, t in ts.items():
            print(n.scope, LayerProfiler.avg_time(t))

    @staticmethod
    def flatten(ts):
        if isinstance(ts, (list, tuple, set)):
            yield from chain(*[LayerProfiler.flatten(t) for t in ts])
        elif isinstance(ts, dict):
            yield from chain(*[LayerProfiler.flatten(t) for t in ts.values()])
        elif isinstance(ts, slice):
            yield from LayerProfiler.flatten(ts.start)
            yield from LayerProfiler.flatten(ts.stop)
            yield from LayerProfiler.flatten(ts.step)
        else:
            yield ts

    @staticmethod
    def only_tensors_that_require_grad(ts):
        return [t for t in LayerProfiler.flatten(ts) if isinstance(t, Tensor) and t.requires_grad]

    @staticmethod
    def only_tensors_with_grad_fn(ts):
        return [t for t in LayerProfiler.flatten(ts) if isinstance(t, Tensor)and(t.grad_fn is not None)]

    @staticmethod
    def get_grads(ts):
        return [torch.randn_like(t) for t in ts]

    @staticmethod
    def detach_tensors(ts):
        def detach_if_tensor(t):
            if isinstance(t, Tensor):
                return t.clone().detach().requires_grad_(t.requires_grad)
            return t

        return nested_map(detach_if_tensor, ts)

    @staticmethod
    def avg_time(times):
        max_v = max(times)
        total = sum([t for t in times if t < max_v])
        return total / (len(times) - 1)

    @staticmethod
    def should_profile(node, function, args, kwargs, output=None):
        if node.type is not NodeTypes.LAYER:
            return False
        if output is None:
            tmp_arg, tmp_kwargs = LayerProfiler.detach_tensors((args, kwargs))
            output = function(*tmp_arg, **tmp_kwargs)
            del tmp_arg
            del tmp_kwargs
        return len(LayerProfiler.only_tensors_that_require_grad(output)) > 0 or len(LayerProfiler.only_tensors_with_grad_fn(output))
