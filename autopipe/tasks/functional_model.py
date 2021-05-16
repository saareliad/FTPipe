import torch

from pipe.models.transformers_utils import resize_token_embeddings
from . import register_task
from .new_t5 import T5Partitioner, ParsePartitioningT5Opts, TiedT5ForConditionalGeneration, T5Config, T5Tokenizer

import torch.nn.functional as F

_MODEL_DIM = 10000

class FunctionalModel(torch.nn.Module):
    def __init__(self):
        super(FunctionalModel, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(_MODEL_DIM, _MODEL_DIM))
        self.w2 = torch.nn.Parameter(torch.randn(_MODEL_DIM, _MODEL_DIM))
        self.w3 = torch.nn.Parameter(torch.randn(_MODEL_DIM, _MODEL_DIM))
        self.w4 = torch.nn.Parameter(torch.randn(_MODEL_DIM, _MODEL_DIM))
        self.w5 = torch.nn.Parameter(torch.randn(_MODEL_DIM, _MODEL_DIM))

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1))
        x = F.relu(F.linear(x, self.w2))
        x = F.relu(F.linear(x, self.w3))
        x = F.relu(F.linear(x, self.w4))
        x = F.relu(F.linear(x, self.w5))
        x = F.dropout(F.linear(x, self.w1))
        return x


class DumTFunctionalModelPartitioner(T5Partitioner):

    def get_model(self, args) -> torch.nn.Module:
        return FunctionalModel()


    def get_input(self, args, analysis=False):
        if analysis:
            return torch.randn( args.analysis_batch_size ,_MODEL_DIM)

        return torch.randn(args.partitioning_batch_size, _MODEL_DIM)


register_task("functional_model", ParsePartitioningT5Opts, DumTFunctionalModelPartitioner)
