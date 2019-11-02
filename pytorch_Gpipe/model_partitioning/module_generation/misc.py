from typing import List, Tuple, Dict, Set
from torch.nn import Module
tab = '    '
dtab = tab + tab


def generateMiscMethods():
    state_dict = generateStateDictFunction()
    named_parameters = generateNamedParametersFunction()
    named_buffers = generateNamedBuffersFunction()

    return "\n".join([state_dict, named_parameters, named_buffers]) + "\n\n"


def generateStateDictFunction():
    state_dict_function = ["def state_dict(self):",
                           f"# we return the state dict of this part as it should be in the original model",
                           "state = super().state_dict()",
                           f"lookup = self.lookup",
                           "result = dict()",
                           "for k, v in state.items():",
                           f"{tab}if k in lookup:",
                           f"{dtab}result[lookup[k]] = v",
                           f"{tab}else:",
                           f"{dtab}assert '.' in k",
                           f"{dtab}split_idx = k.find('.')",
                           f"{dtab}new_k = lookup[k[:split_idx]] + k[split_idx:]",
                           f"{dtab}result[new_k] = v",
                           f"return result"]

    return f"{tab}" + f"\n{dtab}".join(state_dict_function)


def generateNamedParametersFunction():
    named_parameters_function = ["def named_parameters(self):",
                                 f"# we return the named parameters of this part as it should be in the original model",
                                 "params = super().named_parameters()",
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
    named_buffers_function = ["def named_buffers(self):",
                              f"# we return the named buffers of this part as it should be in the original model",
                              "params = super().named_buffers()",
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
