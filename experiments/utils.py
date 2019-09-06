from torch import optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pytorch_Gpipe import pipe_model
import argparse
import sample_models
import torchvision
import sys


def kwargs_string(*pos_strings, **kwargs):
    return ', '.join(list(pos_strings) + [f'{key}={val}' for key, val in kwargs.items()])


def reset_mex_memory_allocated():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_max_memory_allocated(i)


def get_max_memory_allocated():
    max_mem = -1

    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.max_memory_allocated(i)
        max_mem = max(max_mem, mem_alloc)

    return max_mem


def call_func_stmt(func, *params, **kwargs):
    if isinstance(func, str):
        func_name = func
    else:
        func_name = func.__name__

    params_str = [str(param) for param in params]

    return f'{func_name}({kwargs_string(*params_str, **kwargs)})'


def train(model, num_classes, num_batches, batch_shape):
    reset_mex_memory_allocated()

    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    batch_size = batch_shape[0]

    one_hot_indices = torch.LongTensor(batch_size).random_(
        0, num_classes).view(batch_size, 1)

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for b in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(*batch_shape)
        labels = torch.zeros(batch_size, num_classes).scatter_(
            1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to(dev))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    print('.', end='')


def plot(means, stds, labels, fig_name, fig_label):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(fig_label)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


def create_pipeline(model, batch_shape, **kwargs):
    return pipe_model(model, sample_batch=torch.randn(*batch_shape), **kwargs)


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
    def __init__(self, *args, uses_dataset=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.uses_dataset = uses_dataset

        models = [
            'AlexNet', 'alexnet', 'DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'GoogLeNet',
            'Inception3', 'inception_v3', 'LeNet', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'resnet152', 'SqueezeNet', 'squeezenet1_0', 'squeezenet1_1', 'VGG', 'vgg11', 'vgg11_bn', 'vgg13',
            'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'WideResNet', 'AmoebaNet_D', 'amoebanetd',
            'resnet101', 'torchgpipe_resnet101'
        ]

        self.add_argument('--model', '-m', help='The model we want to run the experiment on.', choices=models,
                          required=True, dest='model_class')
        self.add_argument('--devices', '-d', help='The number of devices to use in the experiment.', type=int,
                          required=True, dest='num_devices')
        self.add_argument('--classes', '-c', help='The number of classes in the prediction problem.', type=int,
                          required=True, dest='num_classes')
        self.add_argument('--model_params', help='The parameters for the model', nargs='*', action=StoreDict,
                          default={})
        self.add_argument('--pipeline_params', help='Parameters for the pipeline itself other then devices', nargs='*',
                          action=StoreDict, default={})

        if uses_dataset:
            self.add_argument('--dataset', '-s', choices=list(torchvision.datasets.__all__), required=True)
            self.add_argument('--ds_root', '-r', type=str, required=True)
        else:
            self.add_argument('--batch_shape', '-s', help='The shape of one batch.', nargs='*', type=int, required=True)
            self.add_argument('--tests_config', help='Any other config kwargs for the test', nargs='*',
                              action=StoreDict, default={})

    def parse_args(self, *args, **kwargs):
        res = vars(super().parse_args(*args, **kwargs))

        res['model_params']['num_classes'] = res['num_classes']

        res['model_class'] = getattr(sys.modules['sample_models'], res['model_class'])

        if self.uses_dataset:
            ds_class = getattr(sys.modules['torchvision.datasets'], res['dataset'])
            res['dataset'] = ds_class(res['ds_root'])

        return res
