import torch.nn as nn
import torch


class treeNet(nn.Module):
    def __init__(self, depth):
        super(treeNet, self).__init__()

        if depth == 0:
            self.left = nn.Linear(10, 10)
            self.right = nn.ReLU()
        else:
            self.left = treeNet(depth-1)
            self.right = treeNet(depth-1)

        self.middle = nn.Sigmoid()

    def forward(self, x):
        x = self.left(x)
        x = self.middle(x)
        x = self.right(x)
        return x


class combinedTreeNet(nn.Module):
    def __init__(self, depth):
        super(combinedTreeNet, self).__init__()

        if depth == 0:
            self.left = nn.Linear(10, 10)
            self.right = nn.ReLU()
        else:
            self.left = treeNet(depth-1)
            self.right = combinedTreeNet(depth-1)

    def forward(self, x):
        x = self.left(x)
        x = self.right(x)
        return x


class arithmeticNet(nn.Module):
    def __init__(self):
        super(arithmeticNet, self).__init__()

        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.l1(x) + self.l2(x)


class netWithTensors(nn.Module):

    def __init__(self):
        super(netWithTensors, self).__init__()

        self.l1 = nn.Linear(10, 10)
        self.register_buffer('buff', torch.randn(10))
        self.param = nn.Parameter(torch.randn(10))
        self.l2 = nn.Linear(10, 10)

    def forward(self, x):
        left = self.l1(x) * self.buff
        right = self.l2(x) * self.param

        return left + right
