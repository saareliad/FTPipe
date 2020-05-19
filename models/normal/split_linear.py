""" Implementation of split linear layers. This will be automatically partitioned by our algorithm.

For two layers one after another,
prefer a combination of SplitLinear and SplitLinearIn,  to make just a single all-reduce.
(The idea follows MegatronLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitLinear(nn.Module):
    """ Split Linear layer.
        by the dimention of out_features
        (For each split, the output will be smaller. Requires stack at the end)
    """

    __constants__ = ['in_features', 'out_features']

    def __init__(self, other: nn.Linear, n_split: int):
        super().__init__()

        self.in_features = other.in_features
        self.out_features = other.out_features

        weight = other.weight
        bias = other.bias

        t = out_features // n_split
        if out_features % n_split != 0:
            raise NotImplementedError()

        self.weights = nn.ParameterList(
            [nn.Parameter(a.contiguous()) for a in weight.split(t, dim=0)])

        self.n_split = n_split

        if bias is None:
            self.biases = [None] * n_split
        else:
            self.biases = nn.ParameterList(
                [nn.Parameter(a.contiguous()) for a in bias.split(t, dim=0)])

    def forward(self, input):
        return torch.cat(
            [F.linear(input, w, b) for w, b in zip(self.weights, self.biases)],
            dim=-1)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.biases[0] is not None)


class SplitLinearIn(nn.Module):
    """ Split Linear layer.
        by the dimention of in_features
        (For each split, the input will be smaller.
        Requires sum and adding bias at the end)
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, other: nn.Linear, n_split: int):
        super().__init__()

        self.in_features = other.in_features
        self.out_features = other.out_features

        weight = other.weight
        self.bias = other.bias

        t = in_features // n_split
        self.t = t
        if in_features % n_split != 0:
            raise NotImplementedError()

        self.weights = nn.ParameterList(
            [nn.Parameter(a.contiguous()) for a in weight.split(t, dim=1)])

        self.n_split = n_split

    def forward(self, input):
        # TODO: assert?
        si = input.split(self.t, dim=-1)
        r = torch.stack([F.linear(i, w) for w, i in zip(self.weights, si)],
                        0).sum(0)

        if self.bias is not None:
            return r.add_(self.bias)
        else:
            return r

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


#############################################################
# Same classes, without the reduce operation.
# This is effective for stacking layers one after another.
#############################################################


class NoReduceSplitLinear(SplitLinear):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, input):
        return [
            F.linear(input, w, b) for w, b in zip(self.weights, self.biases)
        ]
        # returns self.n_split outputs
        # reduece is stack


class NoReduceSplitLinearIn(SplitLinearIn):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, split_input):
        # split_input is iteratable of tensor inputs.
        # required reduce is torch.stack(...,0).sum(0).add_(biase)
        return [F.linear(i, w) for w, i in zip(self.weights, split_input)]


if __name__ == "__main__":
    in_features = 10
    out_features = 10
    bias = True
    n_split = 2

    # Test
    batch = 3

    lin = nn.Linear(in_features, out_features, bias)
    print("-I- testing split linear classes...")
    for split_lin_cls in [SplitLinear, SplitLinearIn]:
        split_lin = split_lin_cls(lin, n_split)
        # print(split_lin)

        x = torch.randn(batch, in_features)
        a = lin(x)
        b = split_lin(x)

        assert torch.allclose(a, b), (a.shape, b.shape)
        print("    OK", split_lin_cls.__name__)

    print()
    print("-I- testing 2 stacked split linear classes...")
    # TODO: sometimes this fails, probabaly numerical...

    first = NoReduceSplitLinear(lin, n_split)
    second = NoReduceSplitLinearIn(lin, n_split)
    if second.bias is not None:
        a = torch.stack(second(first(x)), 0).sum(0).add_(second.bias)
    else:
        a = torch.stack(second(first(x)), 0).sum(0)
    b = lin(lin(x))

    assert torch.allclose(a, b), (a.shape, b.shape)
    print("    OK", first.__class__.__name__, "-->", second.__class__.__name__,
          "-->", "reduce")
