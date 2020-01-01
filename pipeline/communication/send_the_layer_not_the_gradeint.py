import torch
from torch.nn import Module


def get_innermost_last_layer(partition: Module):
    list_children = list(partition.children())
    if not list_children:
        return partition
    last_layer = list_children[-1]
    del list_children
    return get_innermost_last_layer(last_layer)


def get_innermost_first_layer(partition: Module):
    list_children = list(partition.children())
    if not list_children:
        return partition
    last_layer = list_children[0]
    del list_children
    return get_innermost_last_layer(last_layer)


class DummyLayerHelper(Module):
    """
    Helper class for implementing the
    "Don't send the gradient, use a dummy layer" trick for model parallelism.

    This class helps the sender/reciever of the gradients to sync a dummy layer
    and perform backward pass.


    TODO:
        Find a way to do the backward in reciever.
        Implement the agreement on the layer given partitioning Config.
        Compute stats - (how much communication we spare).

        Option/Warning to turn off when the tradeoff not worth it.
    """

    def __init__(self, sender_partition: Module, is_sender):
        # Get last layer
        # Requires that the layer is registered last.
        self.is_sender = is_sender
        self.layer = get_innermost_first_layer(sender_partition)

    def create_grad_buffers_for_rcv(self):
        assert self.is_sender
        # TODO check on batch norm, were we need to create buffers for buffers too (yo dawg).
        state_dict = self.layer.state_dict(keep_vars=True)
        for k, v in state_dict.items():
            if v.requires_grad and (v.grad is None):
                v.grad = torch.zeros_like(v.data).requires_grad_(False)

    def tensors_to_sync(self):
        # Get the state dict (parameters, buffers)
        state_dict = self.layer.state_dict(keep_vars=True)

        # could be "safer" to use grad.data, but cpp code already does this.
        grads_dict = {f'{k}_grad': v.grad for k,
                      v in state_dict.items() if v.requires_grad}

        return [v.data for v in state_dict.values()] + list(grads_dict.values())

    def recompute_and_backwards(self, x):
        # TODO...
        assert not self.is_sender

        # Build the comutation graph.
        out = self.layer(x)
        raise NotImplementedError()

        # TODO: and now? sum()? 1? ...

    # def inplace_sync_from_parameters(self, rcev_parameters):
    #     # TODO: this fucntion is not needed, as we rcv writes stright to the parmeters.
    #     with torch.no_grad():
    #         rcev_parameters = list(rcev_parameters)
    #         total_params = len(self.layer.parameters())
    #         total_buffers = len(self.layer.buffers())

    #         grads = rcev_parameters[total_params + total_buffers:]
    #         buffers = rcev_parameters[total_params:total_params + total_buffers]
    #         parameters = rcev_parameters[:total_params]

    #         for cur_p, real_p in zip(self.layer.parameters(), parameters):
    #             assert cur_p.shape == real_p.shape
    #             cur_p.data = real_p.data

    #         for cur_p, real_p in zip(self.layer.buffers(), buffers):
    #             assert cur_p.shape == real_p.shape
    #             cur_p.data = real_p.data

    #         for cur_p, real_p in zip(filter(lambda p: p.requires_grad, self.layer.parameters()), grads):
    #             cur_p.grad.data = real_p


if __name__ == "__main__":
    # Unit test
    # python -m torch.distributed.launch --nproc_per_node 2 send_the_layer_not_the_gradeint.py
    BACKAND = "gloo"
    CUDA = False
    import torch.distributed as dist

    if BACKAND != "mpi":
        import argparse
        parser = argparse.ArgumentParser(description='Test')

        parser.add_argument('--rank', default=None,
                            type=int, help="Rank of worker")
        parser.add_argument('--local_rank', default=0,
                            type=int, help="Local rank of worker")
        parser.parse_args()

    def wait(handlers):
        for i in handlers:
            i.wait()

    dist.init_process_group(BACKAND, init_method="env://", world_size=2)
    print("INIT DONE")
    device = torch.device(
        f"cuda:{dist.get_rank()}" if CUDA and BACKAND == 'mpi' else "cpu")
    if dist.get_rank() == 0:

        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10), torch.nn.Linear(10, 1))

        ll = DummyLayerHelper(model, is_sender=True)
        x = torch.randn(1, 5)
        model(x).sum().backward()
        tensors = ll.tensors_to_sync()

        print(f"Printing sent items out of {len(tensors)} send items")
        for i in tensors:
            print(i)

        handlers = [dist.isend(p, 1, tag=i+1) for i, p in enumerate(tensors)]

    else:
        # Note: there is an agreement about the dummy model.
        dummy_model = torch.nn.Linear(10, 1)

        ll = DummyLayerHelper(dummy_model, is_sender=False)
        ll.create_grad_buffers_for_rcv()  # create buffers for recv
        tensors = ll.tensors_to_sync()

        handlers = [dist.irecv(p, 0, tag=i+1) for i, p in enumerate(tensors)]

    wait(handlers)

    if dist.get_rank() == 0:
        pass
    else:
        print("Recved:")
        for p in ll.layer.parameters():
            print(p)
            print("Gradinet:")
            print(p.grad)
