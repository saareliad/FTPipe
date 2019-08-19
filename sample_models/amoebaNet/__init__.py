from collections import OrderedDict
from typing import Tuple, Optional, List

import torch
from torch import nn
from torch import Tensor
from .genotype import Genotype, amoebanetd_genotype
from .utils import Conv_3x3, Conv_7x1_1x7, Conv_Cell, FactorizedReduce, Pool_Operation

__all__ = ["AmoebaNet_D"]

# based on torchGpipe implementation https://github.com/kakaobrain/torchgpipe/blob/master/examples/amoebanet/__init__.py


def conv_1x1(channel: int, stride: int, affine: bool) -> Conv_Cell:
    return Conv_Cell(channel, channel, 1, stride, 0, affine, use_relu=True)


def avg_pool_3x3(channel: int, stride: int, affine: bool) -> Pool_Operation:
    return Pool_Operation('avg', 3, channel, stride, affine)


def max_pool_2x2(channel: int, stride: int, affine: bool) -> Pool_Operation:
    return Pool_Operation('max', 2, channel, stride, affine)


def skip_connect(channel: int, stride: int, affine: bool) -> nn.Module:
    if stride == 1:
        return nn.Identity()
    return FactorizedReduce(channel, channel, affine)


class Classifier(nn.Module):
    def __init__(self, channel_prev: int, num_classes: int):
        super().__init__()

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(channel_prev, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        s1 = self.global_pooling(x)
        y = self.classifier(s1.view(s1.size(0), -1))
        return y


class Stem(nn.Module):
    def __init__(self, channel: int):
        super(Stem, self).__init__()
        self.conv_cell = Conv_Cell(3, channel, 3, 2, 1, False)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        out = self.conv_cell(x)
        return out


OPS = {
    'skip_connect': skip_connect,
    'avg_pool_3x3': avg_pool_3x3,
    'max_pool_2x2': max_pool_2x2,
    'conv_7x1_1x7': Conv_7x1_1x7,
    'conv_1x1____': conv_1x1,
    'conv_3x3____': Conv_3x3,
}


# TODO there is a big difference between the traced layers and what is expected
class AmoebaNet_D(nn.Module):
    """an AmoebaNet-D model for ImageNet."""

    def __init__(self, num_classes: int = 10, num_layers: int = 4,
                 num_filters: int = 512,
                 genotype: Optional[Genotype] = None):
        super(AmoebaNet_D, self).__init__()

        genotype = amoebanetd_genotype if genotype is None else genotype
        assert isinstance(genotype, Genotype)
        channel = num_filters // 4

        channel_prev_prev, channel_prev, channel_curr = channel, channel, channel
        cells = []

        # reduction
        channel_curr *= 2
        reduction_prev = False
        reduction = True
        cell = Amoeba_Cell(genotype, channel_prev_prev,
                           channel_prev, channel_curr, reduction, reduction_prev)
        multiplier = len(cell.concat_indices)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        # reduction
        channel_curr *= 2
        reduction_prev = True
        reduction = True
        cell = Amoeba_Cell(genotype, channel_prev_prev,
                           channel_prev, channel_curr, reduction, reduction_prev)
        multiplier = len(cell.concat_indices)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        # not reduction
        reduction_prev = True
        reduction = False
        for _ in range(num_layers):
            cell = Amoeba_Cell(genotype, channel_prev_prev,
                               channel_prev, channel_curr, reduction, reduction_prev)
            multiplier = len(cell.concat_indices)
            channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
            cells.append(cell)
            reduction_prev = False

        # reduction
        channel_curr *= 2
        reduction_prev = False
        reduction = True
        cell = Amoeba_Cell(genotype, channel_prev_prev,
                           channel_prev, channel_curr, reduction, reduction_prev)
        multiplier = len(cell.concat_indices)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        # not reduction
        reduction_prev = True
        reduction = False
        for _ in range(num_layers):
            cell = Amoeba_Cell(genotype, channel_prev_prev,
                               channel_prev, channel_curr, reduction, reduction_prev)
            multiplier = len(cell.concat_indices)
            channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
            cells.append(cell)
            reduction_prev = False

        # reduction
        channel_curr *= 2
        reduction_prev = False
        reduction = True
        cell = Amoeba_Cell(genotype, channel_prev_prev,
                           channel_prev, channel_curr, reduction, reduction_prev)
        multiplier = len(cell.concat_indices)
        channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
        cells.append(cell)

        # not reduction
        reduction_prev = True
        reduction = False
        for _ in range(num_layers):
            cell = Amoeba_Cell(genotype, channel_prev_prev,
                               channel_prev, channel_curr, reduction, reduction_prev)
            multiplier = len(cell.concat_indices)
            channel_prev_prev, channel_prev = channel_prev, multiplier * channel_curr
            cells.append(cell)
            reduction_prev = False

        self.stem = Stem(channel)
        self.twin = Twin()
        self.cells = nn.Sequential(*cells)
        self.classifier = Classifier(channel_prev, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.stem(x)
        out = self.cells(self.twin(out))
        return self.classifier(out[1])


class Amoeba_Cell(nn.Module):
    def __init__(self, genotype: Genotype,
                 channel_prev_prev: int, channel_prev: int, channel: int,
                 reduction: bool, reduction_prev: bool):
        super(Amoeba_Cell, self).__init__()

        preprocess0 = nn.Sequential()
        if reduction_prev:
            preprocess0 = FactorizedReduce(channel_prev_prev, channel)
        elif channel_prev_prev != channel:
            preprocess0 = Conv_Cell(channel_prev_prev, channel, 1, 1, 0, True)

        preprocess0: nn.Module = preprocess0
        preprocess1 = Conv_Cell(channel_prev, channel, 1, 1, 0, True)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        ops = []
        for name, index in zip(op_names, indices):
            if reduction and index < 2:
                stride = 2
            else:
                stride = 1
            op = OPS[name](channel, stride, True)
            ops.append((op, index))

        layers = []
        indices = []

        layers = [
            InputOne(preprocess0, i=0),
            TwinLast(),
            InputOne(preprocess1, i=1),
            # Output: (preprocess0(x[0]), preprocess1(x[1]), x[1])
            # The last tensor x[1] is passed until the cell output.
            # The comments below call x[1] "skip".
        ]

        for i in range(len(ops) // 2):
            op0, i0 = ops[i*2]
            op1, i1 = ops[i*2 + 1]

            layers.extend([
                InputOne(op0, i=i0, insert=2+i),
                # Output: x..., op0(x[i0]), skip]

                InputOne(op1, i=i1, insert=2 + i + 1),
                # Output: x..., op0(x[i0]), op1(x[i1]), skip

                MergeTwo(2 + i, 2 + i + 1),
                # Output: x..., op0(x[i0]) + op1(x[i1]), skip
            ])

        layers.extend([
            # Move skip to the head.
            Shift(),
            # Output: skip, x...

            FirstAnd(Concat(concat)),
            # Output: skip, concat(x...)
        ])

        self.layers = nn.Sequential(*layers)
        self.indices = indices
        self.concat_indices = concat

    def forward(self, xs):
        out = self.layers(xs)
        return out


Tensors = Tuple[Tensor, ...]


class Hack(nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Twin(Hack):
    """Duplicates the tensor::
         ┌──────┐
     a --│ Twin │--> a
         │   '--│--> a
         └──────┘
    """

    def forward(self, tensor: torch.Tensor) -> Tensors:  # type: ignore
        return tensor, tensor


class TwinLast(Hack):
    """Duplicates the last tensor::
        a -----> a
        b -----> b
        c --+--> c
            +--> c'
    """

    def forward(self, tensors: Tensors) -> Tensors:  # type: ignore
        return tensors + (tensors[-1],)


class InputOne(Hack):
    """Picks one tensor for the underlying module input::
        a -----> a
        b --f--> f(b)
        c -----> c
    """

    def __init__(self, module: nn.Module, i: int, insert: Optional[int] = None):
        super().__init__()
        self.module = module
        self.i = i
        self.insert = insert

    def forward(self, tensors: Tensors) -> Tensors:  # type: ignore
        i = self.i

        # for t in tensors:
        #     print(t.shape[1:])
        # print("\n")
        input = tensors[i]
        output = self.module(input)

        if not isinstance(output, tuple):
            output = (output,)

        if self.insert is None:
            # Replace with the input.
            return tensors[:i] + output + tensors[i+1:]

        return tensors[:self.insert] + output + tensors[self.insert:]


class Shift(Hack):
    """Moves the last tensor ahead of the tensors::
            +--> c
        a --|--> a
        b --|--> b
        c --+
    """

    def forward(self, tensors: Tensors) -> Tensors:  # type: ignore
        return (tensors[-1],) + tensors[:-1]


class MergeTwo(Hack):
    """Merges the last two tensors and replace them with the result::
        a -----> a
        b --+--> b+c
        c --+
    """

    def __init__(self, i: int, j: int):
        super().__init__()
        self.i = i
        self.j = j

    def forward(self, tensors: Tensors) -> Tensors:  # type: ignore
        i = self.i
        j = self.j

        # Set the initial value as the first tensor
        # to type as 'Tensor' instead of 'Union[Tensor, int]'.
        merged = sum(tensors[i+1:j+1], tensors[i])

        return tensors[:i] + (merged,) + tensors[j+1:]


class FirstAnd(Hack):
    """Skips the first tensor, executes the underlying module by the remaining
    tensors::
        a -----> a
        b --+--> f(b, c)
        c --+
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, tensors: Tensors) -> Tensors:  # type: ignore
        output = self.module(tensors[1:])
        if not isinstance(output, tuple):
            output = (output,)
        return (tensors[0],) + output


class Concat(Hack):
    """Concat all tensors::
        a --+
        b --+--> concat(a, b, c)
        c --+
    """

    def __init__(self, indices: List):
        super().__init__()
        self.indices = indices

    def forward(self, tensors: Tensors) -> torch.Tensor:  # type: ignore
        return torch.cat([tensors[i] for i in self.indices], dim=1)
