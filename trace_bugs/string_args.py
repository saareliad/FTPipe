import torch
import torch.nn as nn
import torch.nn.functional as F


class StringArg(nn.Module):
    def __init__(self):
        super(StringArg, self).__init__()
        self.l = nn.LogSoftmax(dim=1)

    def forward(self, x, y):
        logits = self.l(x)
        return F.nll_loss(logits, y, reduction='mean')


if __name__ == "__main__":
    x = torch.randn(10, 4)
    y = torch.randint(4, (10,))
    sample = (x, y)
    model = StringArg()

    print("traced")
    print(torch.jit.trace(model, sample).graph)

    print("scripted")
    print(torch.jit.script(model).graph)
