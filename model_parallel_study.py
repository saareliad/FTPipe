from models.normal import resnet50
from models.partitioned.resnet50.resnet50 import ModelParallel, layerDict, tensorDict


import torch
import torch.nn as nn
import torch.optim as optim

num_classes = 1000
num_batches = 10
batch_size = 256
image_w = 224
image_h = 224
num_repeat = 2


def get_model_parallel(num_chunks=4):
    base_model = resnet50()
    return ModelParallel(layerDict(base_model), tensorDict(base_model), CPU=False, num_chunks=num_chunks)


def train(model, version="regular"):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)
    if version == "regular":
        for _ in range(num_batches):
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                .scatter_(1, one_hot_indices, 1)

            # run forward pass
            optimizer.zero_grad()
            outputs = model(inputs.to('cuda:0'))

            # run backward pass
            labels = labels.to(outputs.device)
            loss_fn(outputs, labels).backward()
            optimizer.step()
    elif version == "pipeline":
        for _ in range(num_batches):
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                .scatter_(1, one_hot_indices, 1)

            # run forward pass
            optimizer.zero_grad()
            outputs = model.pipelined_forward(inputs)

            # run backward pass
            labels = labels.to(outputs.device)
            loss_fn(outputs, labels).backward()
            optimizer.step()
    else:
        for _ in range(num_batches):
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                .scatter_(1, one_hot_indices, 1)

            # run forward pass
            optimizer.zero_grad()
            outputs = model.pipelined_forward_with_streams(inputs)

            # run backward pass
            labels = labels.to(outputs.device)
            loss_fn(outputs, labels).backward()
            optimizer.step()


import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


def compare_generated_model_parallel():
    # use modelParallel without pipelining inputs
    if False:
        stmt = "train(model,'regular')"
        model_parallel_setup = "model=get_model_parallel()"
        mp_run_times = timeit.repeat(
            stmt, model_parallel_setup, number=1, repeat=num_repeat, globals=globals())
        mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)
        print("finished simple model parallel")

        single_gpu_setup = "model = resnet50().to('cuda:0')"
        rn_run_times = timeit.repeat(
            stmt, single_gpu_setup, number=1, repeat=num_repeat, globals=globals())
        rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

        print("finished single gpu")
        # compare to inputPipelining with 32 chunks with and without streams
        setup = "model = get_model_parallel(num_chunks=32)"
        stmt = "train(model,'pipeline')"
        pp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

        print("finished pipeline 32 chunks")
        stmt = "train(model,'streams')"
        pp_streams_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        pp_streams_mean, pp_streams_std = np.mean(
            pp_streams_run_times), np.std(pp_streams_run_times)
        print("finished pipeline with streams 32 chunks")
        plot([mp_mean, rn_mean, pp_mean, pp_streams_mean],
             [mp_std, rn_std, pp_std, pp_streams_std],
             ['Model Parallel 4GPUS', 'Single GPU',
              'Pipelining', 'Pipelining with streams'],
             'mp_vs_rn_vs_pp_vs_pp_streams.png')

    # plot influence of number of chunks
    if False:
        pp_means = []
        pp_stds = []
        pp_streams_means = []
        pp_streams_stds = []
        num_chunks = [4, 6, 8, 10, 12, 14, 16, 18, 20]

        print("chunk num effects")
        for c in num_chunks:
            setup = f"model = get_model_parallel(num_chunks={c})"
            stmt = "train(model,'pipeline')"
            pp_run_times = timeit.repeat(
                stmt, setup, number=1, repeat=num_repeat, globals=globals())
            pp_means.append(np.mean(pp_run_times))
            pp_stds.append(np.std(pp_run_times))
            print(f"finished pipeline {c} chunks")

            stmt = "train(model,'streams')"
            pp_streams_run_times = timeit.repeat(
                stmt, setup, number=1, repeat=num_repeat, globals=globals())
            pp_streams_means.append(np.mean(pp_streams_run_times))
            pp_streams_stds.append(np.std(pp_streams_run_times))
            print(f"finished pipeline streams {c} chunks")

        fig, ax = plt.subplots()
        ax.errorbar(num_chunks, pp_means, yerr=pp_stds, label='pipelining')
        ax.errorbar(num_chunks, pp_streams_means,
                    yerr=pp_streams_stds, label='pipelining with streams')
        ax.set_ylabel('ResNet50 Execution Time (Second)')
        ax.set_xlabel('Pipeline num Chunks')
        ax.set_xticks(num_chunks)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig("effects of number of chunks")
        plt.close(fig)


if __name__ == "__main__":
    compare_generated_model_parallel()
