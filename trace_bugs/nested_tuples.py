import torch
import torch.nn as nn


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.l = Nested()

    def forward(self, x):
        a, (b, c) = self.l(x)
        return a + b + c


class Nested(nn.Module):
    def forward(self, x):
        return (x * 3, (x * 4, x * 5))


if __name__ == "__main__":
    sample = torch.randn(150, 100)
    traced = torch.jit.trace(module(), sample)
    print("traced")
    print(str(traced.graph))

    # check that the traces are correct
    test_sample = torch.randn(100, 100)
    expected = module()(test_sample)
    actual = traced(test_sample)

    assert torch.allclose(expected, actual)
    print("scripted")
    print(torch.jit.script(module()).graph)
