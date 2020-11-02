import torch


class PartitionRngStasher:
    """
    Utility class to stash and restore RNG state.
    Used during recomputation.

    Pop happens when we restore the state (therefore can only restore state once).

    # NOTE: 
    #   (1) it will be problematic when 2 recomputing stages are use the same device. (e.g tied GPipe)
    #   (2) currently does not set numpy or python random seeds. just pytorch's. (TODO)
    """

    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.state = {}

        # devices list for `fork_rng` method
        self.devices = [self.device] if self.device.type == 'cuda' else []

    def stash_rng_state(self, micro_batch_index):
        """ Stash RNG state """
        cpu_rng_state = torch.get_rng_state()
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                gpu_rng_state = torch.cuda.get_rng_state()
        else:
            gpu_rng_state = None

        self.state[micro_batch_index] = (cpu_rng_state, gpu_rng_state)

    def restore_rng_state(self, micro_batch_index):
        cpu_rng_state, gpu_rng_state = self.state.pop(micro_batch_index)
        torch.set_rng_state(cpu_rng_state)
        if not (gpu_rng_state is None):
            torch.cuda.set_rng_state(gpu_rng_state, self.device)

    def clear_state(self):
        self.state.clear()


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # CPU
    model = nn.Sequential(nn.Linear(10, 10), nn.Dropout2d(0.5))
    s = PartitionRngStasher()
    s.stash_rng_state(0)
    a = model(torch.ones(4, 10))
    b = model(torch.ones(4, 10))
    assert not torch.allclose(a, b)
    with torch.random.fork_rng(devices=s.devices):
        s.restore_rng_state(0)
        c = model(torch.ones(4, 10))
    assert torch.allclose(a, c)

    # Cuda
    model = nn.Sequential(nn.Linear(10, 10), nn.Dropout2d(0.5))
    device = torch.device("cuda:0")
    x = torch.ones(4, 10, device=device)
    model = model.to(device)
    s = PartitionRngStasher(device=device)
    s.stash_rng_state(0)
    a = model(x)
    with torch.random.fork_rng(devices=s.devices):
        s.restore_rng_state(0)
        b = model(x)
    torch.allclose(a, b)
