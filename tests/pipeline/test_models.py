
import torch
import torch.nn as nn
import torch.nn.functional as functional


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, *hidden_dims: int):
        super(MLP, self).__init__()

        dims = (in_dim, *hidden_dims, out_dim)
        windowed_dims = ((dims[i], dims[i+1]) for i in range(len(dims) - 1))

        self.layers = (nn.Linear(in_d, out_d) for in_d, out_d in windowed_dims)

    def forward(self, x):
        for layer in self.layers:
            x = functional.relu(layer(x))

        return x
