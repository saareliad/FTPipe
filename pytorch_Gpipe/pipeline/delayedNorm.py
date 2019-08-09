from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import batch_norm as BatchNorm
from torch.nn.modules.batchnorm import Module, _BatchNorm


class DelayedBatchNorm(_BatchNorm):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: Optional[float] = 0.1, affine: bool = True, track_running_stats: bool = True, num_micro_batches: int = 1):
        super().__init__(num_features, eps=eps, momentum=momentum,
                         affine=affine, track_running_stats=track_running_stats)

        # mean => sum(micro_batches) / num_micro_batches
        # var =>  sum(micro_batches^2) / num_micro_batches - mean^2

        if self.track_running_stats:
            self.register_buffer('num_micro_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
            self.register_buffer('recorded_micro_batches_size',
                                 torch.tensor(0, dtype=torch.long))
            self.num_micro_batches_in_batch = num_micro_batches
            self.register_buffer("running_micro_sum",
                                 torch.zeros_like(self.running_mean))
            self.register_buffer("running_micro_sum_squares",
                                 torch.zeros_like(self.running_var))

    def _check_input_dim(self, x: Tensor):
        if x.dim() <= 2:
            raise ValueError(
                f'expected at least 3D input but input has only {x.dim()}D')

    def reset_running_stats(self):
        super().reset_running_stats()
        if self.track_running_stats:
            self._reset_running_stats()

    def _reset_running_stats(self):
        if hasattr(self, 'running_micro_sum'):
            self.running_micro_sum.zero_()
            self.running_micro_sum_squares.zero_()
            self.recorded_micro_batches_size.zero_()
            self.num_micro_batches_tracked.zero_()

    def _record_mini_batch(self):
        ''' update statistics for the minibatch'''

        # taken from the implementation of _batchNorm.forward
        exponential_average_factor = 0.0
        if self.momentum != None:  # use exponential moving average
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / \
                    float(self.num_batches_tracked)

            # calculate stats of the micro batch
            mean = self.running_micro_sum / self.recorded_micro_batches_size
            variance = self.running_micro_sum_squares / self.recorded_micro_batches_size
            variance -= (mean**2)

            # update running mean
            self.running_mean *= 1 - exponential_average_factor
            self.running_mean += mean * exponential_average_factor

            # update running_var
            self.running_var *= 1 - exponential_average_factor
            self.running_var += variance * exponential_average_factor

            self._reset_running_stats()

    def _record_micro_batch(self, x: Tensor):
        '''gather statistics for the micro_batch'''
        dim = [0]
        dim.extend(range(2, x.dim()))

        # TODO do some of those operations actually require no_grad?
        with torch.no_grad():
            self.running_micro_sum += x.sum(dim)
            self.running_micro_sum_squares += (x**2).sum(dim)

        size = x.size().numel() // x.size(1)
        self.recorded_micro_batches_size += size
        self.num_micro_batches_tracked += 1

    def _is_recomputing(self):
        # TODO need to know if it's a recomputation or not
        raise NotImplementedError()

    def forward(self, x: Tensor):
        if not self.training:
            # Don't train parameters on the evaluation mode.
            return BatchNorm(
                x,
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=0.0,  # set momentum to 0 because we handle it ourselves
                eps=self.eps,
            )

        if not self._is_recomputing() and self.track_running_stats:
            # Track a micro batch on the training mode
            # but not under a recomputation.
            self._record_micro_batch(x)

            # Update the running statistics for a mini batch
            # if it has tracked enough micro batches.
            if self.num_micro_batches_tracked == self.num_micro_batches_in_batch:
                self._record_mini_batch()

        return BatchNorm(
            x,
            running_mean=None,
            running_var=None,
            weight=self.weight,
            bias=self.bias,
            training=True,
            momentum=0.0,  # set momentum to 0 because we handle it ourselves
            eps=self.eps,
        )

    @classmethod
    def convert(cls, module: Module, num_micro_batches: int = 1) -> Module:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DelayedBatchNorm`::
            eg.
            from torchvision.models.resnet import resnet101
            from pytorchGpipe import pipe_model,DelayedBatchNorm
            model = resnet101()
            model = pipe_model(model,microbatch_size,sample_batch,...)
            model = DelayedBatchNorm.convert(model)
        """

        module_output = module

        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DelayedBatchNorm(module.num_features,
                                             module.eps,
                                             module.momentum,
                                             module.affine,
                                             num_micro_batches)

            # use the original buffers and parameters
            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked

        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert(child, num_micro_batches))

        del module
        return module_output
