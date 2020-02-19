import torch
import torch.nn as nn
from pytorch_Gpipe import Pipeline
from pytorch_Gpipe.pipeline.pipeline import SyncBuffersMode, SyncParametersMode


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
        x = self.l(x)
        return (x, x + 1)


if __name__ == "__main__":
    config = {
        'model inputs': ['input0'],
        'model outputs': ['output1', 'output0', 'output2'],
        0: {
            'inputs': ['input0'],
            'outputs': ['output2'],
            'model': Stage0(),
            'ranks': [torch.device('cpu') for _ in range(1)],
            'replicas': [Stage0().share_memory() for _ in range(1)],
            'optimizers': []
        },
        1: {
            'inputs': ['input0'],
            'outputs': ['t0'],
            'model': Stage1(),
            'ranks': [torch.device('cpu') for _ in range(2)],
            'replicas': [Stage1().share_memory() for _ in range(2)],
            'optimizers': []
        },
        2: {
            'inputs': ['t0'],
            'outputs': ['output0', 'output1'],
            'model': Stage2(),
            'ranks': [torch.device('cpu') for _ in range(4)],
            'replicas': [Stage2().share_memory() for _ in range(4)],
            'optimizers': []
        }
    }

    config[0]['optimizers'].append(torch.optim.SGD(
        config[0]['replicas'][0].parameters(), 100))

    model = Pipeline(config, output_device='cpu', buffer_sync=SyncBuffersMode.DISABLED,
                     parameter_sync=SyncParametersMode.DISABLED)

    sample = torch.randn(240, 100)

    o0, o1, o2 = model(sample, num_chunks=4)
    loss = o0.norm() + o1.norm() + o2.norm()

    grads = torch.autograd.grad([loss], [o0, o1, o2])

    print("master before backward")
    print(model._shards[0][0].l.weight)
    model.backward(grads)
    print("master after backward")
    print(model._shards[0][0].l.weight)
