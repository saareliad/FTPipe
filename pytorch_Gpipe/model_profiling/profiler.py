from collections import defaultdict
import torch
from torch import Tensor
from .tracer import NodeTypes
from ..utils import detach_tensors, flatten, move_tensors, set_grad_mode, inplace_arithmetic_ops, ExecTimes,force_out_of_place


class GraphProfiler():
    def __init__(self, recomputation=False, n_iter=10, force_no_recomp_scopes=None, profile_ops=True, save_memory_mode=False):
        self.forward_times = defaultdict(list)
        self.backward_times = defaultdict(list)
        self.recomputation = recomputation
        assert n_iter > 0
        self.n_iter = n_iter + 1

        if force_no_recomp_scopes is None:
            self.force_no_recomp_scopes = lambda s: False
        else:
            self.force_no_recomp_scopes = force_no_recomp_scopes

        self.profile_ops = profile_ops
        self.save_memory_mode = save_memory_mode

    def time_forward(self, node, function, args, kwargs):
        if self.save_memory_mode:
            function, args, kwargs = move_tensors((function, args, kwargs),
                                                  'cuda')

        with force_out_of_place(function):
            if self.should_profile(node, function, args, kwargs):
                recomputation = self.recomputation and (
                    not self.force_no_recomp_scopes(node.scope))

                for _ in range(self.n_iter):
                    args, kwargs = detach_tensors((args, kwargs))
                    with torch.set_grad_enabled(not recomputation):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(device='cuda')

                        start.record()
                        function(*args, **kwargs)
                        end.record()

                        torch.cuda.synchronize(device='cuda')
                        self.forward_times[node].append(start.elapsed_time(end))

                # NOTE we do not move the the inputs to the cpu
                # because the graph executor stores them on the cpu anyway
                # using save_memory_mode = True should be used with the model and initial inputs on the cpu


            # so we explicilty set grad to False to avoid inplace operations to leaf tensors
            return set_grad_mode((args, kwargs), False)

    def time_backward(self, node, function, args, kwargs, output):
        with force_out_of_place(function):
            if self.should_profile(node, function, args, kwargs, output=output):
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

            if self.save_memory_mode:
                # NOTE we move the function and output to cpu
                # in order to clear the gpu
                # args and kwargs are just temporaries
                # the graph executor saves the originals on the cpu
                function, output = move_tensors((function, output), 'cpu')

            # detach output from history and start recording again for future operations
            return set_grad_mode(output, True)

    def backward_no_recomputation(self, node, function, args, kwargs, output):
        for _ in range(self.n_iter):
            torch.cuda.synchronize(device='cuda')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            tensors = self.only_tensors_that_require_grad(output)
            grads = GraphProfiler.get_grads(tensors)

            torch.cuda.synchronize(device='cuda')
            start.record()
            torch.autograd.backward(tensors=tensors,
                                    grad_tensors=grads,
                                    retain_graph=True)
            end.record()
            torch.cuda.synchronize(device='cuda')

            self.backward_times[node].append(start.elapsed_time(end))

            GraphProfiler.delete_grads(node, function, (args, kwargs))

    def backward_recomputation(self, node, function, args, kwargs, output):
        for _ in range(self.n_iter):
            args, kwargs = detach_tensors((args, kwargs))
            with torch.enable_grad():
                torch.cuda.synchronize(device='cuda')
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device='cuda')

                start.record()
                output = function(*args, **kwargs)
                tensors = self.only_tensors_that_require_grad(output)
                grads = GraphProfiler.get_grads(tensors)  # FIXME: memory allocation time will be recorded
                torch.autograd.backward(tensors=tensors,
                                        grad_tensors=grads)
                end.record()

                torch.cuda.synchronize(device='cuda')
                self.backward_times[node].append(start.elapsed_time(end))

                GraphProfiler.delete_grads(node, function, (args, kwargs))

    def get_weights(self):
        weights = dict()
        for node, f_times in self.forward_times.items():
            f_time = GraphProfiler.avg_time(f_times)
            if node in self.backward_times:
                b_time = GraphProfiler.avg_time(self.backward_times[node])
            else:
                b_time = 0

            weights[node.scope] = ExecTimes(f_time, b_time)

        return weights

    def print_times(self, backward=False):
        if backward:
            ts = self.backward_times
        else:
            ts = self.forward_times
        for n, t in ts.items():
            print(n.scope, GraphProfiler.avg_time(t))

    @staticmethod
    def only_tensors_that_require_grad(ts):
        return [t for t in flatten(ts) if isinstance(t, Tensor) and t.requires_grad]

    @staticmethod
    def only_tensors_with_grad_fn(ts):
        return [t for t in flatten(ts) if isinstance(t, Tensor)and(t.grad_fn is not None)]

    @staticmethod
    def delete_grads(node, function, ts):
        if node.type is NodeTypes.LAYER:
            for p in function.parameters():
                p.grad = None
        for p in GraphProfiler.only_tensors_that_require_grad(ts):
            p.grad = None

    @staticmethod
    def get_grads(ts):
        return [torch.randn_like(t) for t in ts]

    @staticmethod
    def avg_time(times):
        max_v = max(times)
        total = sum([t for t in times if t < max_v])
        return total / (len(times) - 1)

    def should_profile(self, node, function, args, kwargs, output=None):
        if node.type not in [NodeTypes.LAYER, NodeTypes.OP]:
            return False

        if not self.profile_ops and node.type is NodeTypes.OP:
            return False

        if node.type is NodeTypes.OP:
            # we cannot profile inplace ops
            op_path = node.scope.rsplit("/", maxsplit=1)[1].rsplit("_",maxsplit=1)[0]
            namespace, func_name = op_path.split("::")
            if func_name in inplace_arithmetic_ops:
                return False

        if output is None:
            tmp_arg, tmp_kwargs = detach_tensors((args, kwargs))
            output = function(*tmp_arg, **tmp_kwargs)
            del tmp_arg
            del tmp_kwargs

        output_w_grads = GraphProfiler.only_tensors_that_require_grad(output)
        output_w_grad_fn = GraphProfiler.only_tensors_with_grad_fn(output)

        should_profile = len(output_w_grads) > 0 or len(output_w_grad_fn) > 0

        return should_profile
