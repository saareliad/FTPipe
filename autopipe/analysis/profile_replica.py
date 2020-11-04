import torch

from autopipe.autopipe import move_tensors
from autopipe.autopipe.utils import flatten


def cuda_computation_times(model, inputs):
    """ measure forward/backward time of a partition on the GPU
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    model.cuda()
    # now we move inputs to GPU
    inputs = move_tensors(inputs, 'cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device='cuda')
    start.record()
    outputs = model(*inputs)

    # TODO: can generate targets beforehand to use cross_entropy...
    # TODO: replace randn_like with pre-generated tensors
    # loss = sum((F.cross_entropy(o, torch.randn_like(o)) for o in filter(
    #     lambda t: isinstance(t, torch.Tensor) and t.requires_grad,
    #     flatten(outputs))))

    loss = sum((o.norm() for o in filter(
        lambda t: isinstance(t, torch.Tensor) and t.requires_grad,
        flatten(outputs))))  # FIXME: just use real loss.
    loss.backward()
    end.record()
    torch.cuda.synchronize(device='cuda')
    fb_time = (start.elapsed_time(end))

    return fb_time