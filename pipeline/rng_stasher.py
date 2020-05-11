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
