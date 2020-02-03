import torch
import torch.nn as nn
from typing import Optional


class OptionalInput(nn.Module):
    def __init__(self):
        super(OptionalInput, self).__init__()
        self.l = nn.ReLU()

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.ones(x.shape[1])

        return self.l(x * mask)


if __name__ == "__main__":
    x = torch.randn(10, 4)
    y = torch.randn(4)
    sample = (x, y)
    model = OptionalInput()

    print("traced")
    print(torch.jit.trace(model, sample).graph)

    print("scripted")
    print(torch.jit.script(model).graph)

    print(torch.jit.script(model).code)
