from typing import Tuple
import torch
import torch.nn as nn
tab = '    '
dtab = tab + tab


def generate_partition_state_methods() -> str:
    ''' generate partition methods state_dict() load_state_dict() named_buffers() and named_parameters()
        our custom implementation gurrentees 100% compatibility with the original model same names will be used
    '''
    state_dict = generateStateDictFunction()
    load_state_dict = generateLoadStateDict()
    named_parameters = generateNamedParametersFunction()
    named_buffers = generateNamedBuffersFunction()

    cpu, cuda, to = generateCpuCudaToMethods()

    return "\n".join([state_dict, load_state_dict, named_parameters, named_buffers, cpu, cuda, to]) + "\n\n"


def generateStateDictFunction() -> str:
    '''generates the state_dict function ensuring same keys are used as in the base model
    '''
    state_dict_function = ["def state_dict(self,device=None):",
                           f"# we return the state dict of this part as it should be in the original model",
                           "return state_dict(self,device=device)"]

    return f"{tab}" + f"\n{dtab}".join(state_dict_function)


def generateNamedParametersFunction() -> str:
    ''' generates the named_parameters method ensuring we use the names given to the parametes in the unpartitioned model
    '''
    named_parameters_function = ["def named_parameters(self,recurse=True):",
                                 f"# we return the named parameters of this part as it should be in the original model",
                                 "return named_parameters(self,recurse=recurse)"]
    return f"\n{tab}" + f"\n{dtab}".join(named_parameters_function)


def generateNamedBuffersFunction() -> str:
    ''' generates the named_buffers method ensuring we use the names given to the buffers in the unpartitioned model
    '''
    named_buffers_function = ["def named_buffers(self,recurse=True):",
                              f"# we return the named buffers of this part as it should be in the original model",
                              "return named_buffers(self,recurse=recurse)"]
    return f"\n{tab}" + f"\n{dtab}".join(named_buffers_function)


def generateLoadStateDict() -> str:
    '''generates the load_state_dict method ensures that weights will be assigned to thier correct counterparts inside the partition
    '''
    func = ['def load_state_dict(self, state):',
            "return load_state_dict(self,state)"]

    return f"\n{tab}" + f"\n{dtab}".join(func)


def generateCpuCudaToMethods() -> Tuple[str, str, str]:
    """generates the cpu cuda and to methods of the partitions
       the generated code keeps track of on which device the partition is placed

    Returns:
        Tuple[str, str, str] the generated code
    """
    cpu = f"\n{tab}def cpu(self):\n{dtab}return cpu(self)\n"

    cuda = [f"{tab}def cuda(self,device=None):",
            f"return cuda(self,device=device)\n",
            ]

    to = [f"{tab}def to(self, *args, **kwargs):",
          "return to(self,*args,**kwargs)",
          ]

    return cpu, f"\n{dtab}".join(cuda), f"\n{dtab}".join(to)


# state methods of the partitions to be copied to generated file

def get_state_methods():
    return [state_dict, load_state_dict, named_buffers, named_parameters, cpu, cuda, to]


def state_dict(partition, device=None):
    # we return the state dict of this part as it should be in the original model
    state = nn.Module.state_dict(partition)
    lookup = partition.lookup
    result = dict()
    for k, v in state.items():
        if k in lookup:
            result[lookup[k]] = v if device is None else v.to(device)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            result[new_k] = v if device is None else v.to(device)
    return result


def load_state_dict(partition, state):
    reverse_lookup = {v: k for k, v in partition.lookup.items()}
    device = partition.device
    keys = list(partition.state_dict(None).keys())
    new_state = dict()
    for k in keys:
        if k in reverse_lookup:
            new_state[reverse_lookup[k]] = state[k].to(device)
            continue
        idx = k.rfind(".")
        to_replace = k[:idx]
        if to_replace in reverse_lookup:
            key = reverse_lookup[to_replace] + k[idx:]
            new_state[key] = state[k].to(device)
    nn.Module.load_state_dict(partition, new_state, strict=True)


def named_parameters(partition, recurse=True):
    # we return the named parameters of this part as it should be in the original model
    params = nn.Module.named_parameters(partition, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield (lookup[k], v)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield (new_k, v)


def named_buffers(partition, recurse=True):
    # we return the named buffers of this part as it should be in the original model
    params = nn.Module.named_buffers(partition, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield (lookup[k], v)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield (new_k, v)


def cpu(partition):
    partition.device = torch.device('cpu')
    return nn.Module.cpu(partition)


def cuda(partition, device=None):
    if device is None:
        device = torch.cuda.current_device()
    partition.device = torch.device(device)
    return nn.Module.cuda(partition, partition.device)


def to(partition, *args, **kwargs):
    device = None
    if 'device' in kwargs:
        device = kwargs['device']
    elif 'tensor' in kwargs:
        device = kwargs['tensor'].device
    if args:
        if isinstance(args[0], (torch.device, int, str)):
            device = args[0]
        if torch.is_tensor(args[0]):
            device = args[0].device
    if not (device is None):
        partition.device = torch.device(device)
    return nn.Module.to(partition, *args, **kwargs)
