import torch
import torch.nn as nn
import sys
sys.path.append("../")
from pytorch_Gpipe import Pipeline
from pytorch_Gpipe.pipeline.pipeline import SyncBuffersMode
import torch.multiprocessing as mp
import time


class Stage0(nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.device = torch.device('cpu')
        self.l = nn.Linear(4, 4)

    def forward(self, x):
        return (self.l(x),)


class Stage1(nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.device = torch.device('cpu')
        self.l = nn.Linear(4, 4)

    def forward(self, x):
        return (self.l(x),)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    m0 = Stage0()
    m1, m2 = Stage1().to('cuda:0').share_memory(), Stage1().to('cuda:1').share_memory()
    m1.device = torch.device('cuda:0')
    m2.device = torch.device('cuda:1')
    optimzer0 = torch.optim.SGD(m0.parameters(),
                                lr=1, momentum=1, dampening=0, weight_decay=1e-2, nesterov=True)
    optimzer1 = torch.optim.SGD(m1.parameters(),
                                lr=1, momentum=1, dampening=0, weight_decay=1e-2, nesterov=True)
    optimzer2 = torch.optim.SGD(m2.parameters(),
                                lr=1, momentum=1, dampening=0, weight_decay=1e-2, nesterov=True)
    config = {
        'model inputs': ['input0'],
        'model outputs': ['output0', 'output1'],
        0: {  # p2p
            'inputs': ['input0'],
            'outputs': ['output0'],
            'ranks': [torch.device('cpu')],
            'replicas': [m0],
            'optimizers': [optimzer0]
        },
        1: {  # replicated input p2mp
            'inputs': ['output0'],
            'outputs': ['output1'],
            'ranks': [torch.device('cuda:0'), torch.device('cuda:1')],
            'replicas': [m1, m2],
            'optimizers': [optimzer1, optimzer2]
        },
    }

    model = Pipeline(config, output_device='cpu',
                     buffer_sync=SyncBuffersMode.DISABLED)
    print("built pipeline")
    sample = torch.randn(240, 4)

    o0, o1 = model(sample, num_chunks=4)
    loss = o0.norm() + o1.sum()

    grads = torch.autograd.grad([loss], [o0, o1])
    print("done forward")
    assert (o0.shape == o1.shape)
    assert o0.shape == torch.Size([240, 4])
    assert o0.device.type == 'cpu' and o1.device.type == 'cpu'
    model.backward(grads)
    # model.backward returns before the step so we this explicitly waits for an ack after the step
    model.eval()
    print("done backward")
