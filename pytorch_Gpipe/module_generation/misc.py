from typing import List, Tuple, Dict, Set
from torch.nn import Module
tab = '    '
dtab = tab + tab


def generateMiscMethods():
    state_dict = generateStateDictFunction()
    load_state_dict = generateLoadStateDict()
    named_parameters = generateNamedParametersFunction()
    named_buffers = generateNamedBuffersFunction()

    return "\n".join([state_dict, load_state_dict, named_parameters, named_buffers]) + "\n\n"


def generateStateDictFunction():
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


def generateNamedParametersFunction():
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


def generateNamedBuffersFunction():
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


def generateLoadStateDict():
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
