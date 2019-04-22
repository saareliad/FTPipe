
# %%
import timeit
import torch.nn as nn
import math
from torchvision import datasets, transforms

import torch
import torchvision.models
# import hiddenlayer


def get_cifar_10_data_set():
    print("==> Preparing data.. CIFAR-10")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(
        root="data/cifar10", train=False, download=True, transform=transform_test)
    return train_set, test_set


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x += 100

        return x


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


# hiddenlayer.build_graph(resnet20_cifar(), torch.zeros(1, 3, 32, 32))

class complexNet(nn.Module):
    def __init__(self):
        super(complexNet, self).__init__()
        a = nn.Linear(2, 2)

        self.sub1 = nn.Sequential(
            nn.Sequential(a),
            a, nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2)))

        self.sub2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.sub2(self.sub1(x))


class Wrapper(nn.Module):
    def __init__(self, sub_module):
        super(Wrapper, self).__init__()
        self.module = sub_module

    def forward(self, x):
        if(x.is_cuda):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.module(x)
            end.record()
            torch.cuda.synchronize()

            cost = (start.elapsed_time(end))
        else:
            cost = 1000*timeit.Timer(
                lambda: self.module(x)).timeit(1)

        return self.module(x)


class NetProfiler(nn.Module):
    def __init__(self, module):
        super(NetProfiler, self).__init__()
        self.module = module
        for name, sub_module in self.module._modules.items():
            self.module._modules[name] = Wrapper(sub_module)

    def forward(self, x):
        return self.module(x)


if __name__ == "__main__":
    base_model = resnet20_cifar()
    profiler = NetProfiler(complexNet())

    model = complexNet()

# wrap every individual layer in the model with the Wrapper class
    def wrap(module: nn.Module):
        for key, item in module._modules.items():
            if len(list(item.children())) == 0:
                module._modules[key] = Wrapper(item)
            else:
                wrap(item)
        return module

    model = wrap(model)

    indiv_layers = list(filter(lambda l: len(list(l.children()))
                               == 0, model.modules()))

    for layer in model.named_children():
        print(layer)
