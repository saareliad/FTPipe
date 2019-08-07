
import torch
import torch.nn as nn

from tests.pipeline.test_models import MLP
from pytorch_Gpipe import pipe_model
from tests.pipeline.utils import tensors_almost_equal

# TODO all of those tests are busted
# 1.we have no cpu support in the pipeline whatsoever
# 2.the used test models are so small we probably cannot partition them anyway (switch to alexnet)


def test_linear_one_input_cpu():
    batch = torch.rand(1, 10)
    model = nn.Linear(10, 10)
    piped_model = pipe_model(model, microbatch_size=1,
                             sample_batch=batch, devices=['cpu'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_linear_one_input_two_cpus():
    batch = torch.rand(1, 10)
    model = nn.Linear(10, 10)
    piped_model = pipe_model(model, microbatch_size=1,
                             sample_batch=batch, devices=['cpu', 'cpu'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_linear_one_input_one_gpu():
    batch = torch.rand(1, 10)
    model = nn.Linear(10, 10)
    piped_model = pipe_model(model, microbatch_size=1,
                             sample_batch=batch, devices=['cuda'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_linear_one_input_two_gpus():
    batch = torch.rand(1, 10)
    model = nn.Linear(10, 10)
    piped_model = pipe_model(model, microbatch_size=1,
                             sample_batch=batch, devices=['cuda:0', 'cuda:1'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_linear_single_microbatch():
    batch = torch.rand(10, 10)
    model = nn.Linear(10, 10)
    piped_model = pipe_model(model, microbatch_size=10,
                             sample_batch=batch, devices=['cuda:0', 'cuda:1'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_linear_multiple_microbatches():
    batch = torch.rand(10, 10)
    model = nn.Linear(10, 10)
    piped_model = pipe_model(model, microbatch_size=2,
                             sample_batch=batch, devices=['cuda:0', 'cuda:1'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_2_layers_mlp():
    batch = torch.rand(10, 50)
    model = MLP(50, 10, 25)
    piped_model = pipe_model(model, microbatch_size=2,
                             sample_batch=batch, devices=['cuda:0', 'cuda:1'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_3_layers_mlp():
    batch = torch.rand(10, 100)
    model = MLP(100, 10, 50, 25)
    piped_model = pipe_model(model, microbatch_size=2,
                             sample_batch=batch, devices=['cuda:0', 'cuda:1'])

    assert tensors_almost_equal(model(batch), piped_model(batch))


def test_5_layers_mlp():
    batch = torch.rand(10, 100)
    model = MLP(100, 10, 50, 25)
    piped_model = pipe_model(model, microbatch_size=2,
                             sample_batch=batch, devices=['cuda:0', 'cuda:1'])

    assert tensors_almost_equal(model(batch), piped_model(batch))
