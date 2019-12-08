import torch.nn as nn


class DummyDAG(nn.Module):
    def __init__(self):
        super(DummyDAG, self).__init__()
        self.i0 = nn.Linear(200, 100)
        self.i1 = nn.Linear(200, 100)
        self.i2 = nn.Linear(100, 100)
        self.i3 = nn.Linear(100, 100)
        self.i4 = nn.Linear(100, 100)
        self.i5 = nn.Linear(100, 100)
        self.output0 = nn.Linear(100, 100)

    def forward(self, input0):
        i0, i1 = self.i0(input0), self.i1(input0)
        i2 = self.i2(i1)
        i3, i4 = self.i3(i1), self.i4(i1)
        i5 = self.i5(i2+i3+i4)
        output0 = self.output0(i0+i5)

        return output0
