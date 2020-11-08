from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Dataset for generating graphs on the fly
class Dataset(torch.utils.data.Dataset):

    def __init__(self, n, k, max_samples_num):
        'Initialization'
        self.samples_num = max_samples_num
        self.n = n
        self.node_list = list(range(n))
        self.k = k
        self.edge_dict = {}

        A, B = np.tril_indices(n, k=-1)
        ctr = 0
        for i, j in zip(A, B):
            self.edge_dict[(i, j)] = ctr
            self.edge_dict[(j, i)] = ctr
            ctr += 1

    def __len__(self):
        return self.samples_num

    def __getitem__(self, index):
        X = torch.randint(0, 2, (self.n * (self.n - 1) // 2,)) - 0.5

        if index % 2 == 0:
            ch_nodes = np.random.choice(self.node_list, self.k, replace=False)
            X[[self.edge_dict[c] for c in combinations(ch_nodes, 2)]] = 0.5
            return X, torch.tensor([1.0])

        return X, torch.tensor([0.0])


class Net(nn.Module):

    def __init__(self, n, c):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(n * (n - 1) // 2, 3 * n * (n - 1) // 4)
        self.bn1 = nn.BatchNorm1d(3 * n * (n - 1) // 4)
        self.h1_layer = nn.Linear(3 * n * (n - 1) // 4, c)
        self.bn2 = nn.BatchNorm1d(c)
        self.h2_layer = nn.Linear(c, c // 20)
        self.bn3 = nn.BatchNorm1d(c // 20)
        self.output_layer = nn.Linear(c // 20, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.input_layer(x)))
        x = F.leaky_relu(self.bn2(self.h1_layer(x)))
        x = F.leaky_relu(self.bn3(self.h2_layer(x)))
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    # N = number of nodes in graph, K = clique size
    # C = constant as defined in the paper, samples_num = arbitrary high number
    N, K, C, samples_num = 361, 18, 20000, 1e11
    # Loss, Optimizers etc..
    loss_func = nn.BCEWithLogitsLoss()

    model = Net(N, C)
    dataset = Dataset(N, K, samples_num)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print(f"Num parameters: {sum(p.numel() for p in model.parameters())}")

    try:
        from torchsummary import summary

        print(summary(model, input_size=(N * (N - 1) // 2,)))
    except ImportError as e:
        print("please install torchsummary for additional information")
