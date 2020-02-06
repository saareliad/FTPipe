"""AmoebaNet-D for ImageNet"""

# based on torchGpipe implementation https://github.com/kakaobrain/torchgpipe/blob/master/examples/amoebanet/__init__.py


from collections import OrderedDict
from typing import Iterator, List, Tuple, Union

import torch
from torch import Tensor, nn

from .genotype import (NORMAL_CONCAT, NORMAL_OPERATIONS, REDUCTION_CONCAT,
                       REDUCTION_OPERATIONS)
from .utils import FactorizedReduce

__all__ = ['amoebanetd']


def relu_conv_bn(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 ) -> nn.Module:
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class Classify(nn.Module):

    def __init__(self, channels_prev: int, num_classes: int):
        super().__init__()
        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(channels_prev, num_classes)

    def forward(self, states: Tuple[Tensor, Tensor]) -> Tensor:
        x, _ = states
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(nn.ReLU(inplace=False),
                         nn.Conv2d(3, channels, 3, stride=2,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(channels))


class Cell(nn.Module):
    def __init__(self,
                 channels_prev_prev: int,
                 channels_prev: int,
                 channels: int,
                 reduction: bool,
                 reduction_prev: bool,
                 ):
        super().__init__()

        self.reduce1 = relu_conv_bn(
            in_channels=channels_prev, out_channels=channels)

        self.reduce2: nn.Module = nn.Identity()
        if reduction_prev:
            self.reduce2 = FactorizedReduce(channels_prev_prev, channels)
        elif channels_prev_prev != channels:
            self.reduce2 = relu_conv_bn(
                in_channels=channels_prev_prev, out_channels=channels)

        if reduction:
            self.indices, op_classes = zip(*REDUCTION_OPERATIONS)
            self.concat = REDUCTION_CONCAT
        else:
            self.indices, op_classes = zip(*NORMAL_OPERATIONS)
            self.concat = NORMAL_CONCAT

        self.operations = nn.ModuleList()

        for i, op_class in zip(self.indices, op_classes):
            if reduction and i < 2:
                stride = 2
            else:
                stride = 1

            op = op_class(channels, stride)
            self.operations.append(op)

    def extra_repr(self) -> str:
        return f'indices: {self.indices}'

    def forward(self,
                input_or_states: Union[Tensor, Tuple[Tensor, Tensor]],
                ) -> Tuple[Tensor, Tensor]:
        if isinstance(input_or_states, tuple):
            s1, s2 = input_or_states
        else:
            s1 = s2 = input_or_states

        skip = s1

        s1 = self.reduce1(s1)
        s2 = self.reduce2(s2)

        _states = [s1, s2]

        for i in range(0, len(self.operations), 2):
            h1 = _states[self.indices[i]]
            h2 = _states[self.indices[i + 1]]

            op1 = self.operations[i]
            op2 = self.operations[i + 1]

            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            _states.append(s)

        return torch.cat([_states[i] for i in self.concat], dim=1), skip


def amoebanetd(num_classes: int = 10,
               num_layers: int = 4,
               num_filters: int = 512,
               ) -> nn.Sequential:
    """Builds an AmoebaNet-D model for ImageNet."""
    layers = OrderedDict()

    repeat_normal_cells = num_layers // 3

    channels = num_filters // 4
    channels_prev_prev = channels_prev = channels
    reduction_prev = False

    def make_cells(reduction: bool, channels_scale: int, repeat: int) -> Iterator[Cell]:
        nonlocal channels_prev_prev
        nonlocal channels_prev
        nonlocal channels
        nonlocal reduction_prev

        channels *= channels_scale

        for i in range(repeat):
            cell = Cell(channels_prev_prev,
                        channels_prev,
                        channels,
                        reduction,
                        reduction_prev)

            channels_prev_prev = channels_prev
            channels_prev = channels * len(cell.concat)
            reduction_prev = reduction

            yield cell

    def reduction_cell() -> Cell:
        return next(make_cells(reduction=True, channels_scale=2, repeat=1))

    def normal_cells() -> Iterator[Tuple[int, Cell]]:
        return enumerate(make_cells(reduction=False, channels_scale=1, repeat=repeat_normal_cells))

    # Stem for ImageNet
    layers['stem1'] = Stem(channels)
    layers['stem2'] = reduction_cell()
    layers['stem3'] = reduction_cell()

    # AmoebaNet cells
    layers.update((f'cell1_normal{i+1}', cell) for i, cell in normal_cells())
    layers['cell2_reduction'] = reduction_cell()
    layers.update((f'cell3_normal{i+1}', cell) for i, cell in normal_cells())
    layers['cell4_reduction'] = reduction_cell()
    layers.update((f'cell5_normal{i+1}', cell) for i, cell in normal_cells())

    # Finally, classifier
    layers['classify'] = Classify(channels_prev, num_classes)

    return nn.Sequential(layers)
