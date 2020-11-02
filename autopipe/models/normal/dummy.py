import torch
from torch import Tensor
import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.l0 = nn.Linear(100, 100)
        self.l1 = nn.Linear(100, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 100)

    def forward(self, x):
        output2 = self.l0(x)
        t0 = self.l1(x)
        t1 = self.l2(t0)
        output0, output1 = self.l3(t1)

        return output1, output0, output2


class Stage0(nn.Module):
    def __init__(self, layers, tensors):
        super(Stage0, self).__init__()
        assert 'Dummy/Linear[l0]' in layers
        self.l = layers['Dummy/Linear[l0]']
        assert isinstance(self.l, nn.Linear)

    def forward(self, x):
        return (self.l(x),)


class Stage1(nn.Module):
    def __init__(self, layers, tensors):
        super(Stage1, self).__init__()
        assert 'Dummy/Linear[l1]' in layers
        self.l = layers['Dummy/Linear[l1]']
        assert isinstance(self.l, nn.Linear)

    def forward(self, x):
        return (self.l(x),)


class Stage2(nn.Module):
    def __init__(self, layers, tensors):
        super(Stage2, self).__init__()
        assert 'Dummy/Linear[l2]' in layers
        self.l = layers['Dummy/Linear[l2]']
        assert isinstance(self.l, nn.Linear)

    def forward(self, x):
        return (self.l(x),)


class Stage3(nn.Module):
    def __init__(self, layers, tensors):
        super(Stage3, self).__init__()
        assert 'Dummy/Linear[l3]' in layers
        self.l = layers['Dummy/Linear[l3]']
        assert isinstance(self.l, nn.Linear)

    def forward(self, x):
        x = self.l(x)
        return (x, x + 1)
