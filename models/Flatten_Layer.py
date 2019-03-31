import torch.nn as nn


class FlattenLayer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
