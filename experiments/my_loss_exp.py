import torch
import torch.nn as nn
import time
import numpy as np
# add our code to the path so we could import it
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))

from pytorch_Gpipe import pipe_model
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as transforms
import argparse
from sample_models import alexnet, resnet101, vgg19_bn, squeezenet1_1, densenet201, \
    WideResNet, AmoebaNet_D as amoebanet, amoebanetd as torchgpipe_amoebanet, torchgpipe_resnet101
import platform

MODELS = {
    "alexnet": alexnet,
    "resnet101": resnet101,
    "vgg19_bn": vgg19_bn,
    "squeezenet1_1": squeezenet1_1,
    "densenet201": densenet201,
    "WideResNet": WideResNet,
    "amoebanet": amoebanet,
    "torchgpipe_amoebanet": torchgpipe_amoebanet,
    "torchgpipe_resnet101": torchgpipe_resnet101
}

# setups


def single_gpu(model_class: nn.Module, devices, *model_args, **model_kwargs):
    model = model_class(*model_args, **model_kwargs).to(devices[0])
    used_devices = [devices[0]]

    return model, used_devices


def pipeLine(model_class: nn.Module, devices, pipe_sample, model_args, model_kwargs, pipeline_args, pipeline_kwargs):
    net = model_class(*model_args, **model_kwargs).to(devices[0])
    net(pipe_sample)
    torch.cuda.synchronize()

    piped = pipe_model(net, pipe_sample.shape[0], pipe_sample, *pipeline_args,
                       devices=devices, **pipeline_kwargs)
    return piped, piped.module_devices


def dataParallel(model_class: nn.Module, devices, *model_args, **model_kwargs):
    return nn.DataParallel(model_class(*model_args, **model_kwargs), devices).to(devices[0]), devices


SETUPS = {
    "single_gpu": single_gpu,
    "pipeLine": pipeLine,
    "dataParallel": dataParallel
}

# the experiment itself
img_size = (224, 224)


def create_dataloaders(batch_train, batch_test):
    """
    Assumes the following folder structures:
        cats_dogs/cat/<img>
        cats_dogs/cat/<img>

        ...

        cats_dogs/dog/<img>
        cats_dogs/dog/<img>
    """
    dataset_dir = "cats_dogs"
    tfms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root=dataset_dir, transform=tfms)
    train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])

    train_loader = DataLoader(train_set, batch_size=batch_train, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_test, shuffle=False, num_workers=2)

    return train_loader, test_loader


def loss_exp(config):
    assert torch.cuda.is_available(), "gpus are required"
    devices = [torch.device('cuda', i)
               for i in range(torch.cuda.device_count())]

    setup = config['setup']
    model_class = config['model']
    model_args = config['model_args']
    model_kwargs = config['model_kwargs']
    model_kwargs['num_classes'] = 2
    batch_size = config['batch_size']
    pipeLine_kwargs = config['pipeLine_kwargs']
    pipeLine_args = config['pipeLine_args']
    epochs = config['epochs']
    batch_shape = config['batch_shape']
    microbatch_size = config['microbatch_size']
    profile_sample_size = config['profile_sample_size']
    if microbatch_size is None:
        microbatch_size = batch_size // len(devices)

    if setup is pipeLine:
        assert len(devices) > 1, "automatic partitioning does not work for 1 gpu"
        pipe_sample = torch.randn(
            (profile_sample_size,) + batch_shape[1:]).to(devices[0])
        model, used_devices = pipeLine(model_class, devices, pipe_sample, model_args,
                                       model_kwargs, pipeLine_args, pipeLine_kwargs)
    else:
        model, used_devices = setup(
            model_class, devices, *model_args, **model_kwargs)

    used_devices = list(used_devices)
    in_device = used_devices[0]
    out_device = used_devices[-1]
    batch_size = batch_shape[0]
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loader, test_loader = create_dataloaders(batch_size, batch_size)

    throughputs = []
    elapsed_times = []
    memory_consumptions = {device: [] for device in used_devices}
    losses = []
    accuracies = []

    # run one epoch and gather statistics
    def run_epoch(epoch):
        torch.cuda.synchronize(in_device)
        tick = time.time()

        data_trained = 0
        steps = len(train_loader)
        for i, data in enumerate(train_loader):
            batch, target = data
            data_trained += batch.shape[0]

            output = model(batch.to(in_device))
            gt = target.to(output.device)
            loss = F.cross_entropy(output, gt)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # estimate statistics after each batch
            percent = i / steps * 100
            throughput = data_trained / (time.time() - tick)

            # print every 100 steps
            if i % 100 == 0 or i == steps - 1:
                print(
                    f"{epoch+1}/{epochs} epochs ({percent:.2f}%%) | {throughput:.2f} samples/sec (estimated)")

        torch.cuda.synchronize(in_device)
        tock = time.time()

        # calculate exact statistics after epoch is finished
        test_loss, test_accuracy = test(model, test_loader, used_devices[0])
        losses.append(test_loss)
        accuracies.append(test_accuracy)

        elapsed_time = tock - tick
        throughput = batch_size * steps / elapsed_time
        print(
            f"{epoch+1}/{epochs} epochs | {throughput:.2f} samples/sec, {elapsed_time:.2f} sec/epoch, test_accuracy {test_accuracy:.2f}, test loss {test_loss:.2f}")

        return throughput, elapsed_time

    exp = setup.__name__
    title = f'loss experiment\n config: {exp}\n used_gpus: {len(used_devices)}\n epochs: {epochs}\n'
    print(title)

    gpus = [torch.cuda.get_device_name(device) for device in used_devices]
    print('system information\n python: %s, torch: %s, cudnn: %s, cuda: %s, \ngpus: %s' % (
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        gpus))
    print("\n")
    for epoch in range(epochs):
        for d in used_devices:
            torch.cuda.reset_max_memory_allocated(device=d)

        throughput, elapsed_time = run_epoch(epoch)
        # first epoch is used as a warmup
        if epoch < 1:
            continue

        for d in used_devices:
            memory_consumptions[d].append(
                torch.cuda.max_memory_allocated(device=d))

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    print("\n")
    n = len(throughputs)
    throughput = sum(throughputs) / n
    elapsed_time = sum(elapsed_times) / n

    avg_mem_per_epoch = {device: sum(
        memory_consumptions[device]) / n for device in used_devices}
    # Just use 'min' instead of 'max' for minimum.
    maximum = max(avg_mem_per_epoch, key=avg_mem_per_epoch.get)

    print(
        f'{title} {throughput:.2f} samples/sec\n{elapsed_time:.2f} sec/epoch (average)\n max memory consumption on device: {maximum} {(avg_mem_per_epoch[maximum]/1e6):.2f} MB')
    print(f"final loss {losses[-1]:.2f} final accuracy {accuracies[-1]:.2f}")


def test(model, test_loader, input_device):
    criterion = F.cross_entropy

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


def get_accuracy(outputs, labels):
    _, predictions = torch.max(outputs.data, 1)

    return (predictions == labels).sum().item() / labels.size(0)


class StoreDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kv = {}
        if not isinstance(values, (list,)):
            values = (values,)
        for value in values:
            n, v = value.split('=')
            kv[n] = v
        setattr(namespace, self.dest, kv)


class ExpParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument('--setup', '-s', help='The way to run the model.',
                          choices=SETUPS.keys(),
                          required=True, dest='setup')
        self.add_argument('--model', '-m', help='The model we want to run the experiment on.', choices=MODELS.keys(),
                          required=True, dest='model')
        self.add_argument('--model_args', '-ma', help='additional args passed to the pipeline', nargs='*',
                          default=[])
        self.add_argument('--model_kwargs', '-mkw', help='additional kwargs passed to the model', nargs='*', action=StoreDict,
                          default={})
        self.add_argument('--batch_size', '-b', help='batch size used in the experiment defaults to 64',
                          type=int, dest='batch_size', default=64)
        self.add_argument('--pipeLine_args', '-pa', help='additional args passed to the pipeline', nargs='*',
                          default=[])
        self.add_argument('--pipeLine_kwargs', '-pkw', help='additional kwargs passed to the pipeline', nargs='*',
                          action=StoreDict, default={})
        self.add_argument('--epochs', '-e', help="the number of training epochs used,\nthe first epoch is a warmup and will not effect results",
                          type=int, default=10)
        self.add_argument('--microbatch_size', '-mb', help="micro batch size of the pipeline\nif not given defaults to batch_size/num_devices",
                          type=int, default=None)
        self.add_argument('--profile_sample_size', '-ps', help='size of batch used to partition the model if testing the pipeline',
                          type=int, default=16)

    def parse_args(self, *args, **kwargs):
        res = vars(super().parse_args(*args, **kwargs))
        res['model'] = MODELS[res['model']]
        res['setup'] = SETUPS[res['setup']]
        net = res['model']

        if net.__name__.find("inception") != -1:
            sample_shape = (3, 299, 299)
        elif net.__name__.find("GoogLeNet") != -1:
            sample_shape = (3, 32, 32)
        elif net.__name__.find("LeNet") != -1:
            sample_shape = (3, 32, 32)
        else:
            sample_shape = (3, 224, 224)

        res['batch_shape'] = (res['batch_size'],) + sample_shape

        return res


if __name__ == "__main__":
    # TODO all var/kw args are treated as strings
    # throuput_exp.py - s single_gpu - m amoebanet - -model_args 10 - -model_kwargs kw0 = 1 kw1 = hello

    parser = ExpParser()
    args = parser.parse_args()

    loss_exp(args)
