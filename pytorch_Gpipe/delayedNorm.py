"""Tracks the running statistics per mini-batch instead of micro-batch."""
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

# taken form torchGpipe repo


class DelayedBatchNorm(_BatchNorm):
    """A BatchNorm layer tracks multiple micro-batches to update running
    statistics per mini-batch.
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: bool = True,
                 num_micro_batches: int = 1,
                 is_recomputing: bool = False
                 ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)

        self.register_buffer('sum', torch.zeros_like(self.running_mean))
        self.register_buffer('sum_squares', torch.zeros_like(self.running_var))

        self.counter = 0
        self.tracked = 0
        self.num_micro_batches = num_micro_batches
        self.is_recomputing = is_recomputing

    def _check_input_dim(self, x: Tensor):
        if x.dim() <= 2:
            raise ValueError(
                'expected at least 3D input (got %dD input)' % x.dim())

    def _track(self, x: Tensor) -> bool:
        """Tracks statistics of a micro-batch."""
        # Dimensions except channel. For example, (0, 2, 3) is for BatchNorm2d.
        dim = [0]
        dim.extend(range(2, x.dim()))

        with torch.no_grad():
            self.sum += x.sum(dim)
            self.sum_squares += (x**2).sum(dim)

        size = x.size().numel() // x.size(1)
        self.counter += size
        self.tracked += 1

        return (self.tracked == self.num_micro_batches)

    def _commit(self):
        """Updates the running statistics of a mini-batch."""
        exponential_average_factor = 0.0
        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
            exponential_average_factor = self.momentum

        mean = self.sum / self.counter
        var = self.sum_squares / self.counter - mean**2

        # Calculate the exponential moving average here.
        m = exponential_average_factor

        self.running_mean *= 1 - m
        self.running_mean += mean * m

        self.running_var *= 1 - m
        self.running_var += var * m

        self.sum.zero_()
        self.sum_squares.zero_()
        self.counter = 0
        self.tracked = 0

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        if not self.training:
            # Don't train parameters on the evaluation mode.
            return F.batch_norm(
                x,
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )

        if not self.is_recomputing:
            # Track a micro-batch on the training mode
            # but not under a recomputation.
            tracked_enough = self._track(x)

            # Update the running statistics for a mini-batch
            # if it has tracked enough micro-batches.
            if tracked_enough:
                self._commit()

        # Normalize a micro-batch and train the parameters.
        return F.batch_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=self.weight,
            bias=self.bias,
            training=True,
            momentum=0.0,
            eps=self.eps,
        )

    @classmethod
    def convertBatchNorm(cls, module: nn.Module, num_micro_batches: int = 1) -> nn.Module:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DelayedBatchNorm`::
            from torchvision.models.resnet import resnet101
            from pytorch_Gpipe.delayedNorm import DelayedBatchNorm
            model = resnet101()
            model = DelayedBatchNorm.convertBatchNorm(model)
        """
        if isinstance(module, DelayedBatchNorm) and module.num_micro_batches is num_micro_batches:
            return module

        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DelayedBatchNorm(module.num_features,
                                             module.eps,
                                             module.momentum,
                                             module.affine,
                                             num_micro_batches)
            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.register_buffer('running_mean', module.running_mean)
            module_output.register_buffer('running_var', module.running_var)
            module_output.register_buffer(
                'num_batches_tracked', module.num_batches_tracked)

            return module_output

        for name, child in module.named_children():
            module.add_module(
                name, cls.convertBatchNorm(child, num_micro_batches))

        return module
