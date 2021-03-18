import warnings
from collections import defaultdict
from itertools import chain

import torch
from torch import Tensor

from .tracer import NodeTypes
from .control_flow_graph import Graph, Node
from ..utils import detach_tensors, flatten, move_tensors, set_grad_mode, inplace_arithmetic_ops, ExecTimes, \
    force_out_of_place


class GraphProfiler():
    def __init__(self, recomputation=False, n_iter=10, force_no_recomp_scopes=None, profile_ops=True,
                 save_memory_mode=False):
        self.profile_memory = save_memory_mode # TODO: profile memroy without save memory mode.
        if not save_memory_mode:
            warnings.warn("Will not profile memory (since save_memory_mode=False)")
        self.forward_times = defaultdict(list)
        self.backward_times = defaultdict(list)
        self.forward_mem = defaultdict(list)
        self.backward_mem = defaultdict(list)
        self.recomputation = recomputation
        NUM_OUTLIERS = 2
        assert n_iter > 0
        self.n_iter = n_iter + NUM_OUTLIERS

        self.not_profiled = dict(fwd=list(), bwd=list())

        if force_no_recomp_scopes is None:
            self.force_no_recomp_scopes = lambda s: False
        else:
            self.force_no_recomp_scopes = force_no_recomp_scopes

        self.profile_ops = profile_ops
        self.save_memory_mode = save_memory_mode

    def time_forward(self, node, function, args, kwargs):
        if self.save_memory_mode:
            if self.profile_memory:
                torch.cuda.reset_peak_memory_stats()
                base_mem = torch.cuda.max_memory_allocated()

            function, args, kwargs = move_tensors((function, args, kwargs),
                                                  'cuda')

        with force_out_of_place(function):
            if self.should_profile(node, function, args, kwargs):
                recomputation = self.recomputation and (not self.force_no_recomp_scopes(node.scope))
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
                        if self.profile_memory:
                            peak_usage = torch.cuda.max_memory_allocated()
                            self.forward_mem[node].append(peak_usage - base_mem)
                        self.forward_times[node].append(start.elapsed_time(end))
                # NOTE we do not move the the inputs to the cpu
                # because the graph executor stores them on the cpu anyway
                # using save_memory_mode = True should be used with the model and initial inputs on the cpu
            elif node.value_type is torch.Tensor:
                self.not_profiled['fwd'].append(node)

            return detach_tensors((args, kwargs))

    def time_backward(self, node, function, args, kwargs, output):

        with force_out_of_place(function):
            if self.should_profile(node, function, args, kwargs, output=output):
                recomputation = self.recomputation and (not self.force_no_recomp_scopes(node.scope))
                if not recomputation:
                    self.backward_no_recomputation(node, function,
                                                   args, kwargs,
                                                   output)
                else:
                    self.backward_recomputation(node, function,
                                                args, kwargs,
                                                output)
            elif node.value_type is torch.Tensor:
                self.not_profiled['bwd'].append(node)
            if self.save_memory_mode:
                # NOTE we move the function and output to cpu
                # in order to clear the gpu
                # args and kwargs are just temporaries
                # the graph executor saves the originals on the cpu
                function, output = move_tensors((function, output), 'cpu')

            return detach_tensors(output)

    def backward_no_recomputation(self, node, function, args, kwargs, output):
        for _ in range(self.n_iter):
            torch.cuda.synchronize(device='cuda')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            tensors = self.only_tensors_that_require_grad(output)
            grads = GraphProfiler.get_grads(tensors)
            torch.cuda.synchronize(device='cuda')
            if self.profile_memory:
                torch.cuda.reset_peak_memory_stats()
                base_mem = torch.cuda.max_memory_allocated()
            start.record()
            torch.autograd.backward(tensors=tensors,
                                    grad_tensors=grads,
                                    retain_graph=True)
            end.record()
            torch.cuda.synchronize(device='cuda')

            self.backward_times[node].append(start.elapsed_time(end))
            if self.profile_memory:
                peak_usage = torch.cuda.max_memory_allocated()
                self.backward_mem[node].append(peak_usage - base_mem)
            GraphProfiler.delete_grads(node, function, (args, kwargs))

    def backward_recomputation(self, node, function, args, kwargs, output):
        for _ in range(self.n_iter):
            args, kwargs = detach_tensors((args, kwargs))
            with torch.enable_grad():
                torch.cuda.synchronize(device='cuda')
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                grads = GraphProfiler.pre_get_grads(function, args, kwargs)
                torch.cuda.synchronize(device='cuda')
                if self.profile_memory:
                    torch.cuda.reset_peak_memory_stats()
                    base_mem = torch.cuda.max_memory_allocated()

                start.record()
                output = function(*args, **kwargs)
                tensors = self.only_tensors_that_require_grad(output)
                # grads = GraphProfiler.get_grads(tensors)  # pre exectured to avoid recording memory allocation times
                torch.autograd.backward(tensors=tensors,
                                        grad_tensors=grads)
                end.record()

                torch.cuda.synchronize(device='cuda')
                self.backward_times[node].append(start.elapsed_time(end))
                if self.profile_memory:
                    peak_usage = torch.cuda.max_memory_allocated()
                    self.forward_mem[node].append(peak_usage - base_mem)
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

    def set_max_memory_usage(self, graph: Graph):
        for node in graph.nodes:
            if node not in self.forward_mem or node not  in self.backward_mem:
                continue
            fwd = self.forward_mem[node]
            bwd = self.backward_times[node]
            max_usage = max(max(fwd), max(bwd))
            node.max_memory_bytes = max(node.max_memory_bytes, max_usage)

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
        return [t for t in flatten(ts) if isinstance(t, Tensor) and (t.grad_fn is not None)]

    @staticmethod
    def delete_grads(node, function, ts):
        if node.type is NodeTypes.LAYER:
            for p in function.parameters():
                p.grad = None
        for p in GraphProfiler.only_tensors_that_require_grad(ts):
            p.grad = None

    @staticmethod
    def get_grads(ts):
        # NOTE: these dummy gradients will cause problems:
        # will probably miss-measure everything with sparsity.
        # (e.g dropout, relu, the layer right before giant embedding layers...)
        return [torch.randn_like(t) for t in ts]

    @staticmethod
    def pre_get_grads(function, args, kwargs):
        with torch.enable_grad():
            set_grad_mode((args, kwargs), True)
            output = function(*args, **kwargs)
            output = GraphProfiler.only_tensors_that_require_grad(output)
            set_grad_mode((args, kwargs), False)
            return GraphProfiler.get_grads(output)

    @staticmethod
    def avg_time(times, drop=2):
        vs = times
        # FIXME: this should only drop one
        max_v = None
        for i in range(drop):
            max_v = max(vs)
            vs_cand = [t for t in vs if t < max_v]
            if len(vs_cand) == 0:
                break
            vs = vs_cand
        assert len(vs) > 0, (max_v, times)
        total = sum(vs)
        return total / (len(vs))

    def should_profile(self, node, function, args, kwargs, output=None):
        if node.type not in [NodeTypes.LAYER, NodeTypes.OP]:
            return False

        if not self.profile_ops and node.type is NodeTypes.OP:
            return False

        if node.type is NodeTypes.OP:
            op_path, idx = node.scope.rsplit("/", maxsplit=1)[1].rsplit("_", maxsplit=1)
            namespace, func_name = op_path.split("::")

            inplace_torch_function = ("torch" in namespace) and (func_name[-1] == '_')
            inplace_tensor_function = (namespace == "Tensor") and (func_name[-1] == "_") and (
                not func_name.startswith("__"))
            inplace_tensor_magic = (namespace == "Tensor") and (func_name in inplace_arithmetic_ops)

            if inplace_tensor_magic or inplace_tensor_function or inplace_torch_function:
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

    def _debug_stats(self):
        not_fwd = set(self.not_profiled['fwd'])
        not_bwd = set(self.not_profiled['bwd'])
        not_fwd_not_bwd = not_bwd.intersection(not_fwd)
        print(f"not fwd {len(not_fwd)}")
        print(f"not bwd {len(not_bwd)}")
        print(f"not fwd and bwd {len(not_fwd_not_bwd)}")

        not_fwd_req_grad = sum(n.req_grad for n in not_fwd)
        not_bwd_req_grad = sum(n.req_grad for n in not_bwd)

        print(f"not fwd req_grad {not_fwd_req_grad}")
        print(f"not bwd req_grad {not_bwd_req_grad}")

        for n in chain(not_fwd, not_bwd):
            assert not n.req_grad

        assert not_fwd == not_bwd
        print()
        for n in not_fwd:
            print(n.scope)

# NOTE we do not profile operations which do not require grad
# we do not profile inplace operations (unless if we enforce out of place)
# the result is that for things like computing masks we do not profile neither fwd not bwd
# this is ok as those computations are not substantial
