import torch.nn as nn
import torch


def createConfig(model, cpu=False):
    config = {'model inputs': ['input0'],
              0: {'inputs': ['input0'], 'outputs': ['i0', 'i1']},
              1: {'inputs': ['i1'], 'outputs': ['i2']},
              2: {'inputs': ['i1'], 'outputs': ['i3', 'i4']},
              3: {'inputs': ['i2', 'i3', 'i4'], 'outputs': ['i5']},
              4: {'inputs': ['i0', 'i5'], 'outputs': ['output0']},
              'model outputs': ['output0']
              }
    device = torch.device('cpu') if cpu else torch.device('cuda:0')
    p0 = P0(model.i0, model.i1).to(device)
    device = torch.device('cpu') if cpu else torch.device('cuda:1')
    p1 = P1(model.i2).to(device)
    device = torch.device('cpu') if cpu else torch.device('cuda:2')
    p2 = P2(model.i3, model.i4).to(device)
    device = torch.device('cpu') if cpu else torch.device('cuda:3')
    p3 = P3(model.i5).to(device)
    device = torch.device('cpu') if cpu else torch.device('cuda:4')
    p4 = P4(model.output0).to(device)
    config[0]['model'] = p0
    config[1]['model'] = p1
    config[2]['model'] = p2
    config[3]['model'] = p3
    config[4]['model'] = p4
    return config


class P0(nn.Module):
    def __init__(self, i0, i1):
        super(P0, self).__init__()
        self.i0 = i0
        self.i1 = i1

    def forward(self, input0):
        return (self.i0(input0), self.i1(input0))


class P1(nn.Module):
    def __init__(self, i2):
        super(P1, self).__init__()
        self.i2 = i2

    def forward(self, i1):
        return (self.i2(i1),)


class P2(nn.Module):
    def __init__(self, i3, i4):
        super(P2, self).__init__()
        self.i3 = i3
        self.i4 = i4

    def forward(self, i1):
        return (self.i3(i1), self.i4(i1))


class P3(nn.Module):
    def __init__(self, i5):
        super(P3, self).__init__()
        self.i5 = i5

    def forward(self, i2, i3, i4):
        return (self.i5(i2+i3+i4),)


class P4(nn.Module):
    def __init__(self, output0):
        super(P4, self).__init__()
        self.output0 = output0

    def forward(self, i0, i5):
        return (self.output0(i0+i5),)
