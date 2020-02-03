import torch
import torch.nn as nn


class ABC(nn.Module):
    def __init__(self):
        super(ABC, self).__init__()
        self.l = ReturnTuple()

    def forward(self, x):
        a, b, c = self.l(x)
        return a.relu(), b.sigmoid(), c.softmax(1)


class CBA(nn.Module):
    def __init__(self):
        super(CBA, self).__init__()
        self.l = ReturnTuple()

    def forward(self, x):
        a, b, c = self.l(x)
        return c.relu(), b.sigmoid(), a.softmax(1)


class ReturnTuple(nn.Module):
    def forward(self, x):
        return x * 3, x * 4, x * 5


if __name__ == "__main__":
    sample = torch.randn(10, 10)
    traced_ABC = torch.jit.trace(ABC(), sample)
    traced_CBA = torch.jit.trace(CBA(), sample)

    print("ABC trace")
    print(str(traced_ABC.graph))
    print("CBA trace")
    print(str(traced_CBA.graph))

    print("ABC scripted")
    print(str(torch.jit.script(ABC()).graph))
    print("CBA scripted")
    print(str(torch.jit.script(CBA()).graph))
