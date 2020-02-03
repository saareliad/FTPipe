import torch
import torch.nn as nn


class MultipleSameInput(nn.Module):
    def __init__(self):
        super(MultipleSameInput, self).__init__()
        self.a = A()

    def forward(self, x):
        return self.a(x, x)


class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.r = nn.ReLU()

    # if the same input is fed aka x is y then the trace will show only one input
    # assuming it's the same for output
    def forward(self, x, y):
        return x + 5, self.r(y * 2)


if __name__ == "__main__":
    model = MultipleSameInput()
    sample = torch.randn(10, 100)

    print("traced")
    print(torch.jit.trace(model, sample).graph)

    print("scripted")
    print(torch.jit.script(model).graph)
