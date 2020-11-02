import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List


def copy_attrs(me, other, attr_names: List[str]):
    for name in attr_names:
        setattr(me, name, getattr(other, name))


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

    def forward(self, weight, input):
        return F.linear(input, weight.requires_grad_(), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def pop_weight(self):
        tmp = self.weight
        del self.weight
        return tmp


CLASS_TO_STATELESS_CLASS = {
    nn.Embedding: StatelessEmbedding,
    nn.Linear: StatelessLinear
}


class StatelessSequential(nn.Sequential):
    """Sequential model where first and last layers are tied.
        NOTE: it can be generalized to a model where more layers are tied
    """
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            raise NotImplementedError("not supprting ordered dicts for now")
        else:
            first = args[0]
            last = args[len(args) - 1]
            # first = next(iter(args))
            # last = next(reversed(args))

            supported = [nn.Embedding, nn.Linear]

            def check_supported(l, name):
                if not any([isinstance(l, k) for k in supported]):
                    raise NotImplementedError(
                        f"{name} layer should be one of {supported}, got {l.__class__}"
                    )

            check_supported(first, "first")
            check_supported(last, "last")

            first_cls = CLASS_TO_STATELESS_CLASS[first.__class__]
            last_cls = CLASS_TO_STATELESS_CLASS[last.__class__]

            # TIE, keep for after Module.__init__()
            tied_weight = last.weight.data = first.weight.data

            # instance
            first_stateless_instance = first_cls(first)
            last_stateless_instance = last_cls(last)
            first_stateless_instance.pop_weight()
            last_stateless_instance.pop_weight()

            first = first_stateless_instance
            last = last_stateless_instance

            # init seq
            args = [first, *args[1:-1], last]
            super().__init__(*args)

            self.tied_weight = tied_weight

    def pop_weight(self):
        tmp = self.tied_weight
        del self.tied_weight
        return tmp

    def forward(self, tied_weight, input):
        for i, module in enumerate(self):
            if i == 0 or (i == len(self) - 1):
                input = module(tied_weight, input)
            else:
                input = module(input)
        return input

# NOTE: this is the transparant behavior I wanted at first,
#       but it evoulved to be the other way,
#       this doe not really matter for now.
#       its commented our because there is better design than inheritance,
#       which I'll do when needed...
# class IndependentStatelessSequential(StatelessSequential):
#     def __init__(self, *args):
#         super().__init__(*args)
#         w = super().pop_weight()
#         self.tied_weight = nn.Parameter(w)
#     def pop_weight(self):
#         raise NotImplementedError("its not supposed to be done")
#     def forward(self, input):
#         for i, module in enumerate(self):
#             if i == 0 or (i == len(self) - 1):
#                 input = module(self.tied_weight, input)
#             else:
#                 input = module(input)
#         return input


class CompositionStatelessSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        stateless_seq = StatelessSequential(*args)
        self.tied_w = nn.Parameter(stateless_seq.pop_weight())
        self.stateless_seq = stateless_seq

    def forward(self, *args, **kw):
        return self.stateless_seq.forward(self.tied_w, *args, **kw)
