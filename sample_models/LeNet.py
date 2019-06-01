import torch.nn as nn

__all__ = ["LeNet"]


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.max_pool2d1 = nn.MaxPool2d(2)
        self.max_pool2d2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool2d1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2d2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
