

import torch
from torch import nn
from torch import Tensor


class Pool_Operation(nn.Module):
    def __init__(self, pool_type: str, kernel_size: int, channel: int, stride: int, affine: bool):
        super(Pool_Operation, self).__init__()
        assert pool_type in['avg', 'max']

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride=stride,
                                     padding=1, count_include_pad=False)
        else:
            self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=0)

        self.conv_cell = Conv_Cell(channel, channel, 1,
                                   1, 0, affine, use_relu=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pool(x)
        return self.conv_cell(out)


class Conv_Cell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel, stride, padding, affine: bool, use_relu: bool = True):
        super(Conv_Cell, self).__init__()
        self.relu = nn.ReLU(inplace=False) if use_relu else None
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        out = x if self.relu is None else self.relu(x)
        out = self.conv(out)
        return self.norm(out)


class Conv_3x3(nn.Module):
    def __init__(self, channel: int, stride: int, affine: bool):
        super(Conv_3x3, self).__init__()

        self.conv1_1x1 = Conv_Cell(channel, channel//4, 1,
                                   1, 0, affine)
        self.conv2_3x3 = Conv_Cell(channel//4, channel//4, 3,
                                   (stride, stride), 1, affine)
        self.conv3_1x1 = Conv_Cell(channel//4, channel, 1,
                                   1, 0, affine)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1_1x1(x)
        out = self.conv2_3x3(out)
        return self.conv3_1x1(out)


class Conv_7x1_1x7(nn.Module):
    def __init__(self, channel: int, stride: int, affine: bool):
        super(Conv_7x1_1x7, self).__init__()

        self.conv1_1x1 = Conv_Cell(channel, channel//4, 1,
                                   1, 0, affine)

        self.conv2_1x7 = Conv_Cell(channel//4, channel//4, (1, 7),
                                   (1, stride), (0, 3), affine)

        self.conv3_7x1 = Conv_Cell(channel//4, channel//4, (7, 1),
                                   (stride, 1), (3, 0), affine)

        self.conv4_1x1 = Conv_Cell(channel//4, channel, 1,
                                   1, 0, affine)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1_1x1(x)
        out = self.conv2_1x7(out)
        out = self.conv3_7x1(out)
        return self.conv4_1x1(out)


class FactorizedReduce(nn.Module):
    """Operation Factorized reduce"""

    def __init__(self, in_planes: int, out_planes: int, affine: bool = True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(in_planes, out_planes //
                                2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_planes, out_planes //
                                2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(x)
        branch1 = self.conv_1(x)
        branch2 = self.conv_2(x[:, :, 1:, 1:])
        out = torch.cat([branch1, branch2], dim=1)
        out = self.bn(out)
        return out
