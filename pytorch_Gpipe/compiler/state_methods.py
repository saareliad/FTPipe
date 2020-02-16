from typing import Tuple
tab = '    '
dtab = tab + tab


def generate_state_methods() -> str:
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
    state_dict_function = ["def state_dict(self,device):",
                           f"# we return the state dict of this part as it should be in the original model",
                           "state = super().state_dict()",
                           f"lookup = self.lookup",
                           "result = dict()",
                           "for k, v in state.items():",
                           f"{tab}if k in lookup:",
                           f"{dtab}result[lookup[k]] = v if device is None else v.to(device)",
                           f"{tab}else:",
                           f"{dtab}assert '.' in k",
                           f"{dtab}split_idx = k.find('.')",
                           f"{dtab}new_k = lookup[k[:split_idx]] + k[split_idx:]",
                           f"{dtab}result[new_k] = v if device is None else v.to(device)",
                           f"return result"]

    return f"{tab}" + f"\n{dtab}".join(state_dict_function)


def generateNamedParametersFunction() -> str:
    ''' generates the named_parameters method ensuring we use the names given to the parametes in the unpartitioned model
    '''
    named_parameters_function = ["def named_parameters(self,recurse=True):",
                                 f"# we return the named parameters of this part as it should be in the original model",
                                 "params = super().named_parameters(recurse=recurse)",
                                 f"lookup = self.lookup",
                                 "for k, v in params:",
                                 f"{tab}if k in lookup:",
                                 f"{dtab}yield (lookup[k],v)",
                                 f"{tab}else:",
                                 f"{dtab}assert '.' in k",
                                 f"{dtab}split_idx = k.find('.')",
                                 f"{dtab}new_k = lookup[k[:split_idx]] + k[split_idx:]",
                                 f"{dtab}yield (new_k, v)"]
    return f"\n{tab}" + f"\n{dtab}".join(named_parameters_function)


def generateNamedBuffersFunction() -> str:
    ''' generates the named_buffers method ensuring we use the names given to the buffers in the unpartitioned model
    '''
    named_buffers_function = ["def named_buffers(self,recurse=True):",
                              f"# we return the named buffers of this part as it should be in the original model",
                              "params = super().named_buffers(recurse=recurse)",
                              f"lookup = self.lookup",
                              "for k, v in params:",
                              f"{tab}if k in lookup:",
                              f"{dtab}yield (lookup[k],v)",
                              f"{tab}else:",
                              f"{dtab}assert '.' in k",
                              f"{dtab}split_idx = k.find('.')",
                              f"{dtab}new_k = lookup[k[:split_idx]] + k[split_idx:]",
                              f"{dtab}yield (new_k, v)"]
    return f"\n{tab}" + f"\n{dtab}".join(named_buffers_function)


def generateLoadStateDict() -> str:
    '''generates the load_state_dict method ensures that weights will be assigned to thier correct counterparts inside the partition
    '''
    func = ['def load_state_dict(self, state):',
            'reverse_lookup = {v: k for k, v in self.lookup.items()}',
            'ts = chain(self.named_parameters(), self.named_buffers())',
            'device = list(ts)[0][1].device',
            'keys = list(self.state_dict(None).keys())',
            'new_state = dict()',
            'for k in keys:',
            tab + 'if k in reverse_lookup:',
            dtab + 'new_state[reverse_lookup[k]] = state[k].to(device)',
            dtab + 'continue',
            tab + 'idx = k.rfind(".")',
            tab + 'to_replace = k[:idx]',
            tab + 'if to_replace in reverse_lookup:',
            dtab + 'key = reverse_lookup[to_replace] + k[idx:]',
            dtab + 'new_state[key] = state[k].to(device)',
            'super().load_state_dict(new_state, strict=True)']

    return f"\n{tab}" + f"\n{dtab}".join(func)


def generateCpuCudaToMethods() -> Tuple[str, str, str]:
    """generates the cpu cuda and to methods of the partitions
       the generated code keeps track of on which device the partition is placed

    Returns:
        Tuple[str, str, str] the generated code
    """
    cpu = f"\n{tab}def cpu(self):\n{dtab}self.device=torch.device('cpu')\n{dtab}return super().cpu()\n"

    cuda = [f"{tab}def cuda(self,device=None):",
            f"if device is None:",
            f"{tab}device=torch.cuda.current_device()",
            "self.device=torch.device(device)",
            "return super().cuda(self.device)\n",
            ]

    to = [f"{tab}def to(self, *args, **kwargs):",
          "device = None",
          "if 'device' in kwargs:",
          f"{tab}device = kwargs['device']",
          "elif 'tensor' in kwargs:",
          f"{tab}device = kwargs['tensor'].device",
          "if args:",
          f"{tab}if isinstance(args[0], (torch.device, int, str)):",
          f"{dtab}device = args[0]",
          f"{tab}if torch.is_tensor(args[0]):",
          f"{dtab}device = args[0].device",
          "if not (device is None):",
          f"{tab}self.device = torch.device(device)",
          "return super().to(*args, **kwargs)",
          ]

    return cpu, f"\n{dtab}".join(cuda), f"\n{dtab}".join(to)
