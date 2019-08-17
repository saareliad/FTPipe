from collections import OrderedDict
from typing import Tuple, Optional

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.conv_cell(x)
        return out, out


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

    def __init__(self, num_classes: int = 10,
                 num_layers: int = 1,
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
        self.cells = nn.Sequential(*cells)
        self.classifier = Classifier(channel_prev, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.stem(x)
        out = self.cells(out)
        out = out[0]+out[1]
        return self.classifier(out)


class Amoeba_Cell(nn.Module):
    def __init__(self, genotype: Genotype,
                 channel_prev_prev: int, channel_prev: int, channel: int,
                 reduction: bool, reduction_prev: bool):
        super(Amoeba_Cell, self).__init__()

        preprocess0 = nn.Identity()
        if reduction_prev:
            preprocess0 = FactorizedReduce(channel_prev_prev, channel)
        elif channel_prev_prev != channel:
            preprocess0 = Conv_Cell(channel_prev_prev, channel, 1, 1, 0, True)

        self.preprocess0: nn.Module = preprocess0
        self.preprocess1 = Conv_Cell(channel_prev, channel, 1, 1, 0, True)

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
        for i in range(len(ops) // 2):
            op0, i0 = ops[i*2]
            op1, i1 = ops[i*2 + 1]
            indices.extend([i0, i1])
            layers.extend([op0, op1])

        self.layers = nn.ModuleList(layers)
        self.indices = indices
        self.concat_indices = concat

        self.called = False

    def forward(self, xs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x0, x1 = xs
        branch1 = self.preprocess0(x0)
        branch2 = self.preprocess1(x1)

        out = (branch1, branch2, x1)

        for idx, (op0, op1) in enumerate(zip(self.layers[::2], self.layers[1::2])):
            i0, i1 = self.indices[2*idx], self.indices[2*idx+1]
            i = idx+2
            j = i+1
            # operate on input i0 place result at index i
            out0 = op0(out[i0])
            if not isinstance(out0, tuple):
                out0 = (out0,)
            out = out[:i] + out0 + out[i:]

            # operate on input i1 place result at index j
            out1 = op1(out[i1])
            if not isinstance(out1, tuple):
                out1 = (out1,)
            out = out[:j] + out1 + out[j:]

            # replace indices i,j with their sum (removes one index)
            summed = out[i]+out[j]
            out = out[:i] + (summed,) + out[j+1:]

        # move last output to start
        shifted = (out[-1],)+out[:-1]
        # Output: skip, x...

        stacked = torch.cat([shifted[k]
                             for k in self.concat_indices], dim=1)
        # stacked:concat(x...)

        self.called = True
        # Output: skip, concat(x...)
        return shifted[0], stacked
