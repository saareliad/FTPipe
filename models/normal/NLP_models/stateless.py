import torch.nn as nn
import torch.nn.functional as F


class StatelessEmbedding(nn.Module):

    __constants__ = [
        'num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
        'norm_type', 'scale_grad_by_freq', 'sparse'
    ]

    def __init__(self, other: nn.Embedding):
        super().__init__()

        self.num_embeddings = other.num_embeddings
        self.embedding_dim = other.embedding_dim

        self.padding_idx = other.padding_idx
        self.max_norm = other.max_norm
        self.norm_type = other.norm_type
        self.scale_grad_by_freq = other.scale_grad_by_freq

        self.weight = other.weight

        self.sparse = other.sparse

    def forward(self, weight, input):
        return F.embedding(input, weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq,
                           self.sparse)

    def pop_weight(self):
        tmp = self.weight
        del self.weight
        return tmp

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)


class StatelessLinear(nn.Module):
    """ Stateless Linear layer with shared weight.
        bias is not shared
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, other: nn.Linear):
        super().__init__()

        self.in_features = other.in_features
        self.out_features = other.out_features

        self.weight = other.weight
        self.bias = other.bias

    def forward(self, input, weight):
        # Reversed to solve weird bug
        # print("weight,shape:", weight.shape)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def pop_weight(self):
        tmp = self.weight
        del self.weight
        return tmp
