
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes, Graph
from pytorch_Gpipe.utils import traverse_model, traverse_params_buffs, layerDict, tensorDict
import string
from .partition_forward_method import generate_forward_method, variableNameGenerator
from .partition_init_method import generate_init_method
from .state_methods import get_state_methods, generate_partition_state_methods
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict, deque, defaultdict
import inspect
import os
import pathlib
from .utils import format_shape
tab = '    '
dtab = tab + tab


def create_model_parallel_module(batch_dim: int, name: str, ios: Dict[int, Dict[str,
                                                                                List[str]]],
                                 num_inputs: int,
                                 model_outputs: List[str]) -> str:
    '''create a modelParallel version of the partition config
    '''
    class_decl_and_init = "\n".join([
        f"class ModelParallel(nn.Module):",
        f"{tab}def __init__(self,layers,tensors,CPU=False):",
        f"{dtab}super(ModelParallel,self).__init__()",
        dtab + f"\n{dtab}".join(f"self.stage{i} = Partition{i}(layers,tensors).to('cpu' if CPU else 'cuda:{i}')"
                                for i in ios)
    ])

    model_inputs = [f'input{idx}' for idx in range(num_inputs)]
    forwards = model_parallel_forward(batch_dim, ios,
                                      model_inputs, model_outputs)

    states = f",\n{dtab}{dtab}".join(
        [f"**self.stage{i}.state_dict()" for i in ios])

    states = f"{{{states}}}"

    state_dict = f"\n{dtab}".join(
        ["def state_dict(self):", f"return {states}"])

    loads = f"\n{dtab}".join([f"self.stage{i}.load_state(state)" for i in ios])
    load_state_dict = f"\n{tab}{tab}".join(
        ["def load_state_dict(self,state):", loads])

    buffer_states = f",\n{dtab}{dtab}{tab} ".join(
        [f"self.stage{i}.named_buffers()" for i in ios])
    named_buffers = f"\n{dtab}".join(
        [f"def named_buffers(self):", f"return chain({buffer_states})"])

    parameter_states = f",\n{dtab}{dtab}{tab} ".join(
        [f"self.stage{i}.named_parameters()" for i in ios])
    named_parameters = f"\n{dtab}".join(
        [f"def named_parameters(self):", f"return chain({parameter_states})"])

    parameters = f"\n{dtab}".join([
        "def parameters(self):",
        f"return [p for _,p in self.named_parameters()]"
    ])

    buffers = f"\n{dtab}".join(
        ["def buffers(self):", f"return [b for _,b in self.named_buffers()]"])

    return "\n" + f"\n\n{tab}".join([
        class_decl_and_init, *forwards, state_dict, load_state_dict,
        named_buffers, named_parameters, buffers, parameters
    ]) + "\n\n"


def model_parallel_forward(batch_dim: int, ios: Dict[int, Dict[str, List[str]]],
                           model_inputs: List[str],
                           model_outputs: List[str]) -> List[str]:

    body, activations = forward_statements(ios, model_inputs, model_outputs)
    outputs = ",".join([activations[o] for o in model_outputs])
    forward = simple_forward(model_inputs, body, outputs)

    out_producers = defaultdict(list)
    for idx, io in ios.items():
        for o in io['outputs']:
            if o in model_outputs:
                out_producers[idx].append(activations[o])

    pipe_forward = pipelined_forward(batch_dim, model_inputs,
                                     body, outputs, out_producers)
    return [forward, pipe_forward]


def pipelined_forward(batch_dim: int, model_inputs: List[str], statements: List[str], outputs: str, out_producers: Dict[int, List[str]]) -> str:
    created_upto_i = defaultdict(list)
    created_after_i = defaultdict(list)
    # created_upto_i[idx] = all outputs who are produced by a stage <= idx
    # created_after_i[idx] = all outputs who are producerd by a stage >= idx
    for idx, outs in out_producers.items():
        for i in range(idx + 1):
            created_after_i[i].extend(outs)
        for i in range(idx, len(statements)):
            created_upto_i[i].extend(outs)

    model_outputs = outputs.split(",")
    n_parts = len(statements)
    decleration = f"def pipelined_forward(self,{', '.join(model_inputs)},num_chunks={n_parts}):"
    body = [f"assert num_chunks >= {n_parts}",
            f"batch_dim = {batch_dim}"]

    body.append(f"\n{dtab}# chunk inputs")
    get_inputs = []
    # split inputs
    for i in model_inputs:
        body.extend([f"assert {i}.size(batch_dim) >= num_chunks",
                     f"{i}_chunks = iter({i}.split({i}.size(batch_dim) // num_chunks, dim=batch_dim))"])
        get_inputs.append(f"{i} = next({i}_chunks)")
    body.append(f"\n{dtab}# create output chunk placeholders")

    # create chunk aggregators
    collect_outputs = []
    for o in model_outputs:
        body.append(f"{o}_chunks = []")
        collect_outputs.append(f"{o}_chunks.append({o})")

    # create filling stage
    body.append(f"\n{dtab}# fill the pipeline")
    statements = list(reversed(statements))
    for idx in range(1, n_parts + 1):
        body.extend(get_inputs)
        body.extend(statements[n_parts - idx:n_parts])
        if len(created_upto_i[idx - 1]) > 0:
            for o in created_upto_i[idx - 1]:
                body.append(f"{o}_chunks.append({o})")
        body.append("")

    # create steady stage
    body.append(f"# steady phase")
    body.append(f"for _ in range(num_chunks - {n_parts}):")
    steady = []
    for i in get_inputs:
        steady.append(f"{tab}{i}")
    for s in statements:
        steady.append(f"{dtab}{s}")
    for o in collect_outputs:
        steady.append(f"{dtab}{o}")

    steady = f"\n{tab}".join(steady)
    body.append(steady)

    # create emptying stage
    body.append(f"\n{dtab}# empty the pipeline")
    for idx in range(1, n_parts):
        body.extend(statements[:n_parts - idx])
        if len(created_after_i[idx]) > 0:
            for o in created_after_i[idx]:
                body.append(f"{o}_chunks.append({o})")
        body.append("")

    # cat chunks
    body.append(f"# merge output chunks")
    for o in model_outputs:
        body.append(f"{o} = torch.cat({o}_chunks,dim=batch_dim)")
    body.append("")

    pipelined_forward_function = [decleration] + body + [f"return {outputs}"]

    return f"\n{dtab}".join(pipelined_forward_function)


def simple_forward(model_inputs: List[str], body: List[str], outputs: str) -> str:
    decleration = f"def forward(self,{', '.join(model_inputs)}):"
    forward_function = [decleration] + body + [f"return {outputs}"]
    return f"\n{dtab}".join(forward_function)


def forward_statements(ios: Dict[int, Dict[str, List[str]]],
                       model_inputs: List[str],
                       model_outputs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    '''generates the forward nethod of the model parallel version of the config
    '''
    n_partitions = len(ios)
    arg_gen = variableNameGenerator()

    activations = dict(zip(model_inputs, model_inputs))

    parts = deque(range(n_partitions))

    body = []
    cnt = 0
    while len(parts) > 0:
        if cnt > 3 * n_partitions:
            assert False, "error cycle detected mutual dependecy between generated partitions"
        idx = parts.popleft()

        if all(tensor in activations for tensor in ios[idx]['inputs']):
            inputs = ", ".join(f"{activations[tensor]}"
                               for tensor in ios[idx]['inputs'])
            outputs = []
            for o, t in zip(ios[idx]['outputs'], arg_gen):
                activations[o] = t
                outputs.append(t)

            outputs = ", ".join(outputs)

            body.append(f"{outputs} = self.stage{idx}({inputs})")
            if len(ios[idx]['outputs']) == 1:
                body[-1] += '[0]'
        else:
            cnt += 1
            parts.append(idx)

    return body, activations
