import torch
import torch.nn as nn
from network_profiler import NetProfiler


def test_multi_in_multi_out():
    class multi_model(nn.Module):
        def __init__(self):
            super(multi_model, self).__init__()
            self.path1 = nn.Linear(10, 10)
            self.path2 = nn.Linear(10, 10)
            self.out1 = nn.ReLU()
            self.out2 = nn.Sigmoid()

        def forward(self, in1, in2):
            out1 = self.out1(self.path1(in1))
            out2 = self.out2(self.path2(in2))

            return out1, out2

    model = multi_model()
    profiler = NetProfiler(model, torch.randn(3, 10), torch.randn(5, 10))


if __name__ == "__main__":
    test_multi_in_multi_out()
