import torch
import torch.nn as nn
from pytorch_Gpipe import Pipeline


class Stage0(nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.device = torch.device('cpu')
        self.l = nn.Linear(100, 100)

    def forward(self, x):
        return (self.l(x),)


class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.device = torch.device('cpu')
        self.l = nn.Linear(100, 100)

    def forward(self, x):
        return (self.l(x),)


class Stage2(nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.device = torch.device('cpu')
        self.l = nn.Linear(100, 100)

    def forward(self, x):
        return (self.l(x), x + 1)


config = {
    'model inputs': ['input0'],
    'model outputs': ['output0', 'output1', 'output2'],
    0: {
        'inputs': ['input0'],
        'outputs': ['output2'],
        'model': Stage0(),
        'ranks': [torch.device("cpu")],
        'replicas': [Stage0()],
        'optimizers': []
    },
    1: {
        'inputs': ['input0'],
        'outputs': ['t0'],
        'model': Stage1(),
        'ranks': [torch.device("cpu"), torch.device("cpu")],
        'replicas': [Stage1(), Stage1()],
        'optimizers': []
    },
    2: {
        'inputs': ['t0'],
        'outputs': ['output1', 'output0'],
        'model': Stage2(),
        'ranks': [torch.device("cpu"), torch.device("cpu"), torch.device("cpu"), torch.device("cpu")],
        'replicas': [Stage2(), Stage2(), Stage2(), Stage2()],
        'optimizers': []
    }
}


if __name__ == "__main__":
    Pipeline(config, output_device='cpu')
