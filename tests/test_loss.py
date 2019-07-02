import pytest
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

import torch.optim as optim

from .model_parallel_resnet50 import make_pipeline_resnet

num_classes = 1000
num_batches = 3
batch_size = 120
image_w = 224
image_h = 224

# using constant seed for reproducibility purposes
seed = 42

device_single = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# fixed data-sets
train_set = CIFAR10(root='./data', train=True,
                    download=True, transform=transform)
test_set = CIFAR10(root='./data', train=False,
                   download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def test_resnet50_loss():
    # the models to compare
    model_single = resnet50(num_classes=len(classes)).to(device_single)
    model_pipe = make_pipeline_resnet(
        microbatch_size=20, num_classes=num_classes)

    print(f"Training resnet50 on {device_single}")
    stats_single = train_with_stats_saved(model_single)

    print("Training resnet50 using model-parallelism")
    stats_pipe = train_with_stats_saved(model_pipe)

    plot(stats_single, stats_pipe)

    accuracy_single = test(model_single)
    accuracy_pipe = test(model_pipe)

    print(f"Test accuracy on single device: {accuracy_single}")
    print(f"Test accuracy on model parallel: {accuracy_pipe}")

    # asserts that pipeline's accuracy is within 5% difference from regular model
    assert accuracy_single * 0.95 < accuracy_pipe < accuracy_single * 1.05


def train_with_stats_saved(model):
    # using fixed seed to endure same train-set loader
    torch.manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    total_loss = 0.
    total_accuracy = 0.
    losses = []
    accuracies = []
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backwarod()
        optimizer.step()

        # saving statistics every 25 batches
        total_loss += loss.item()
        total_accuracy += get_accuracy(outputs, labels)
        if i % 25 == 24:
            # average loss so far
            losses.append(total_loss / 25)
            total_loss = 0.

            # average accuracy so far
            accuracies.append(total_accuracy / 25)
            total_accuracy = 0.

    return losses, accuracies


def test(model):
    # always using same test-set loader
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    total_accuracy = 0.
    counter = 0
    with torch.no_grad():
        for data in test_loader:
            counter += 1
            inputs, labels = data
            outputs = model(inputs)

            total_accuracy += get_accuracy(outputs, labels)

    return total_accuracy / counter


def plot(stats_single, stats_pipe):
    loss_s, acc_s = stats_single
    loss_p, acc_p = stats_pipe

    x = range(0, 25 * len(loss_s), 25)

    plt.subplot(1, 2, 1)

    plt.plot(x, loss_s, label='single device')
    plt.plot(x, loss_p, label='model parallel')

    plt.xlabel('batch number')
    plt.ylabel('loss')
    plt.title('MSE Loss Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, acc_s, label='single device')
    plt.plot(x, acc_p, label='model parallel')

    plt.xlabel('batch number')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()


def get_accuracy(outputs, labels):
    _, predictions = torch.max(outputs.data, 1)

    return (predictions == labels).sum().item() / labels.size(0)
