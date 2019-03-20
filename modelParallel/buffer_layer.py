import torch
import torch.nn as nn
import multiprocessing as mp


class BufferLayer(nn.Module):
    def __init__(self, buff_size=2):
        super(BufferLayer, self).__init__()
        self.buf_size = buff_size
        self.idx = 0
        self.input_buffer = torch.zeros(self.buf_size)

    def forward(self, x):
        self.input_buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.buf_size
        return self.input_buffer[self.idx]


buff = BufferLayer(4)
