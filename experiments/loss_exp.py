import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

from sample_models.AlexNet import alexnet
from pytorch_Gpipe import pipe_model


def קספ_loss(model_class, num_devices: int, batch_size: int, model_params: dict, pipeline_params: dict,
             dataset: torch.utils.data.Dataset, train_ratio: float = 0.8, num_epochs=500):
    device_single = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    pipeline_params['devices'] = list(range(num_devices))
    pipeline_params['sample_batch'] = torch.randn(batch_size, *dataset[0].shape)

    train_amount = int(train_ratio * len(dataset))
    test_amount = len(dataset) - train_amount
    train_set, test_set = torch.utils.data.random_split(dataset, (train_amount, test_amount))

    # the models to compare
    model_single = model_class(**model_params).to(device_single)

    model_pipe = pipe_model(model_class(**model_params), **pipeline_params)
    model_pipe.zero_grad()

    model_dp = nn.DataParallel(model_class(**model_params), device_ids=pipeline_params['devices'])

    # train the models
    print(f"Training model on {device_single}")
    stats_single_train, stats_single_test = train_with_stats_saved(model_single, train_set, test_set, num_epochs,
                                                                   batch_size, device_single)

    print("Training model using G-pipe")
    stats_pipe_train, stats_pipe_test = train_with_stats_saved(model_pipe, train_set, test_set, num_epochs, batch_size)

    print("Training model using data parallel")
    stats_dp_train, stats_dp_test = train_with_stats_saved(model_dp, train_set, test_set, num_epochs, batch_size)

    # plot results
    plot_loss(stats_single_train, stats_pipe_train, stats_dp_train)
    plot_loss(stats_single_test, stats_pipe_test, stats_dp_test, False)

    # print(stats_single)
    # print(stats_pipe)
    # print(stats_dp)

    accuracy_single = stats_single_test[1][-1]
    accuracy_pipe = stats_pipe_test[1][-1]
    accuracy_dp = stats_dp_test[1][-1]

    print(f"Test accuracy on single device: {accuracy_single}")
    print(f"Test accuracy on model parallel: {accuracy_pipe}")
    print(f"Test accuracy on data parallel: {accuracy_dp}")

    # asserts that pipeline's accuracy is within 5% difference from regular model
    assert accuracy_single * 0.95 < accuracy_pipe < accuracy_single * 1.05


def train_with_stats_saved(model, train_set, test_set, num_epochs, batch_size, input_device='cpu'):
    # using fixed seed to ensure same train-set loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses, train_accuracies = [], []
    test_stats = []

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accuracies = []

        if epoch % 5 == 0:
            print(f"epoch number {epoch}:")

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs.to(input_device))
            labels = labels.to(outputs.device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # saving statistics every 25 batches
            epoch_losses.append(loss.item())
            epoch_accuracies.append(get_accuracy(outputs, labels))

        # average loss so far
        train_losses.append(np.mean(epoch_losses))

        # average accuracy so far
        train_accuracies.append(np.mean(epoch_accuracies))

        test_stats.append(test(model, test_set, batch_size, input_device))

    return (train_losses, train_accuracies), tuple(zip(*test_stats))


def test(model, test_set, batch_size, input_device='cpu'):
    # always using same test-set loader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    losses, accuracies = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs.to(input_device))
            labels = labels.to(outputs.device)

            loss = criterion(outputs, labels)

            losses.append(loss.item())
            accuracies.append(get_accuracy(outputs, labels))

    return np.mean(losses), np.mean(accuracies)


def plot_loss(stats_single, stats_pipe, stats_dp, training=True):
    loss_s, acc_s = stats_single
    loss_p, acc_p = stats_pipe
    loss_dp, acc_dp = stats_dp

    x = range(0, 25 * len(loss_s), 25)

    plt.subplot(1, 2, 1)

    plt.plot(x, loss_s, label='single device')
    plt.plot(x, loss_p, label='model parallel')
    plt.plot(x, loss_dp, label='data parallel')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'{"train set" if training else "test set"} Cross-Entropy Loss Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, acc_s, label='single device')
    plt.plot(x, acc_p, label='model parallel')
    plt.plot(x, acc_dp, label='data parallel')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()


def get_accuracy(outputs, labels):
    _, predictions = torch.max(outputs.data, 1)

    return (predictions == labels).sum().item() / labels.size(0)
