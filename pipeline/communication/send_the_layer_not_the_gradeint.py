import torch
from torch.nn import Module


def _check_single_tensor(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a single tensor.

    """
    if not isinstance(param, torch.Tensor):
        raise RuntimeError("Invalid function argument. Expected parameter `{}` "
                           "to be of type torch.Tensor.".format(param_name))


class LastLayer():
    def __init__(self, partition: Module):
        # Get last layer
        # Requires that the layer is registered last.

        list_children = list(partition.children())
        if list_children:
            self.last_layer = list(partition.children())[-1]
        else:
            # just a layer given
            self.last_layer = partition

    def create_grad_buffers_for_rcv(self):
        state_dict = self.last_layer.state_dict(keep_vars=True)
        for k, v in state_dict.items():
            if v.requires_grad and (v.grad is None):
                # FIXME: this shouldn't return a parameter, just a tensor
                v.grad = torch.zeros_like(v.data).requires_grad_(False)

    def extract_parameters_to_sync(self):
        # Get the state dict, so
        state_dict = self.last_layer.state_dict(keep_vars=True)

        # TODO: maybe send grad.data
        grads_dict = {f'{k}_grad': v.grad for k,
                      v in state_dict.items() if v.requires_grad}

        return [v.data for v in state_dict.values()] + list(grads_dict.values())


    def inplace_sync_from_parameters(self, rcev_parameters):
        # TODO: this fucntion is not needed, as we rcv stright to the parmeters.
        with torch.no_grad():
            rcev_parameters = list(rcev_parameters)
            total_params = len(self.last_layer.parameters())
            total_buffers = len(self.last_layer.buffers())

            grads = rcev_parameters[total_params + total_buffers:]
            buffers = rcev_parameters[total_params:total_params + total_buffers]
            parameters = rcev_parameters[:total_params]

            for cur_p, real_p in zip(self.last_layer.parameters(), parameters):
                assert cur_p.shape == real_p.shape
                cur_p.data = real_p.data

            for cur_p, real_p in zip(self.last_layer.buffers(), buffers):
                assert cur_p.shape == real_p.shape
                cur_p.data = real_p.data

            for cur_p, real_p in zip(filter(lambda p: p.requires_grad, self.last_layer.parameters()), grads):
                cur_p.grad.data = real_p


if __name__ == "__main__":
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
        ll = LastLayer(model)
        x = torch.randn(1, 5)
        model(x).sum().backward()
        tensors = ll.extract_parameters_to_sync()

        print(f"Printing sent items out of {len(tensors)} send items")
        for i in tensors:
            print(i)

        handlers = [dist.isend(p, 1, tag=i+1) for i, p in enumerate(tensors)]

    else:
        # Note: there is an agreement about the dummy model.
        dummy_model = torch.nn.Linear(10, 1)
        ll = LastLayer(dummy_model)
        ll.create_grad_buffers_for_rcv()  # create buffers for recv
        tensors = ll.extract_parameters_to_sync()

        # grads = [torch.zeros_like(t) for t in tensors if t.requires_gra]

        handlers = [dist.irecv(p, 0, tag=i+1) for i, p in enumerate(tensors)]
        # [dist.irecv(p, 0, tag=len(tensors) + i+1)
        #  for i, p in enumerate(grads)]

    wait(handlers)

    if dist.get_rank() == 0:
        pass
    else:
        print("Recved:")
        for p in ll.last_layer.parameters():
            print(p)
            print(p.grad)
