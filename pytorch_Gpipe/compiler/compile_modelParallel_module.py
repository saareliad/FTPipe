from .partition_forward_method import variableNameGenerator
from ..model_profiling import Graph
from typing import List, Tuple, Dict
from collections import deque, defaultdict

tab = '    '
dtab = tab + tab


def create_model_parallel_module(graph: Graph, batch_dim: int, ios: Dict[int, Dict[str,
                                                                                   List[str]]],
                                 num_inputs: int,
                                 model_outputs: List[str]) -> str:
    '''create a modelParallel version of the partition config
    '''
    class_decl_and_init = "\n".join([
        f"class ModelParallel(nn.Module):",
        f"{tab}def __init__(self,layers,tensors,CPU=False,num_chunks={len(ios)}):",
        f"{dtab}super(ModelParallel,self).__init__()",
        f"{dtab}self.batch_dim = {batch_dim}",
        f"{dtab}self.num_chunks = num_chunks",
        f"{dtab}assert self.num_chunks >= {len(ios)}",
        f"{dtab}self.cpu = CPU",
        f"{dtab}if not CPU:",
        f"{dtab}{tab}# partitions X chunks streams",
        f"{dtab}{tab}self.streams = [[torch.cuda.Stream(f'cuda:{{idx}}') for _ in range(self.num_chunks)] for idx in range({len(ios)})]",
        dtab + f"\n{dtab}".join(f"self.stage{i} = Partition{i}(layers,tensors).to('cpu' if CPU else 'cuda:{i}')"
                                for i in ios)
    ])
    model_inputs = [f'input{idx}' for idx in range(num_inputs)]
    forwards = model_parallel_forward(graph, ios,
                                      model_inputs, model_outputs)

    stream = [f"def stream(self,device_idx,mb_idx):",
              "# return the stream for the current device and micro batch",
              "return torch.cuda.stream(self.streams[device_idx][mb_idx])"]
    stream = f"\n{dtab}".join(stream)

    wait_stream = [f"def wait_stream(self,device_idx,mb_idx):",
                   "stream = self.streams[device_idx][mb_idx]",
                   "# wait until the mb was cleared by previous partition",
                   "stream.wait_stream(self.streams[device_idx-1][mb_idx])",
                   "# wait until previous mb was cleared by this partition",
                   "stream.wait_stream(self.streams[device_idx][mb_idx-1])"]
    wait_stream = f"\n{dtab}".join(wait_stream)

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
        class_decl_and_init, stream, wait_stream,
        *forwards, state_dict, load_state_dict,
        named_buffers, named_parameters, buffers, parameters
    ]) + "\n\n"


def model_parallel_forward(graph: Graph, ios: Dict[int, Dict[str, List[str]]],
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

    created_upto_i = defaultdict(list)
    created_after_i = defaultdict(list)
    # created_upto_i[idx] = all outputs who are produced by a stage <= idx
    # created_after_i[idx] = all outputs who are producerd by a stage >= idx
    for idx, outs in out_producers.items():
        for i in range(idx + 1):
            created_after_i[i].extend(outs)
        for i in range(idx, len(body)):
            created_upto_i[i].extend(outs)

    pipe_forward = pipelined_forward(graph, model_inputs,
                                     body, outputs, created_after_i, created_upto_i, use_streams=False)
    pipe_with_streams = pipelined_forward(graph, model_inputs,
                                          body, outputs, created_after_i, created_upto_i, use_streams=True)

    return [forward, pipe_forward, pipe_with_streams]


def pipelined_forward(graph: Graph, model_inputs: List[str], statements: List[str], outputs: str,
                      created_after_i: Dict[int, List[str]],
                      created_upto_i: Dict[int, List[str]],
                      use_streams: bool) -> str:
    model_outputs = outputs.split(",")
    n_parts = len(statements)
    if use_streams:
        decleration = f"def pipelined_forward_with_streams(self,{', '.join(model_inputs)}):"
    else:
        decleration = f"def pipelined_forward(self,{', '.join(model_inputs)}):"
    get_inputs, body, collect_outputs = generate_get_inputs_splits_and_aggeragators(model_inputs,
                                                                                    model_outputs)
    if use_streams:
        body = ["assert not self.cpu"] + body

    # create filling stage
    body.append(f"\n{dtab}# fill the pipeline")
    statements = list(reversed(statements))
    for idx in range(1, n_parts + 1):
        body.extend(get_inputs)
        for i, s in enumerate(statements[n_parts - idx:n_parts]):
            if use_streams:
                body.append(f"with self.stream({idx-1-i},{i}):")
                body.append(f"{tab}self.wait_stream({idx-1-i},{i})")
                body.append(f"{tab}{s}")
            else:
                body.append(s)
        for o in created_upto_i[idx - 1]:
            body.append(f"{o}_chunks.append({o})")
        body.append("")

    # create steady stage
    body.append(f"# steady phase")
    body.append(f"for idx in range(self.num_chunks - {n_parts}):")
    for i in get_inputs:
        body.append(f"{tab}{i}")
    for i, s in enumerate(statements):
        if use_streams:
            body.append(f"{tab}with self.stream({n_parts-i-1},idx+{i+1}):")
            body.append(
                f"{dtab}self.wait_stream({n_parts-i-1},idx+{i+1})")
            body.append(f"{dtab}{s}")
        else:
            body.append(f"{tab}{s}")
    for o in collect_outputs:
        body.append(f"{tab}{o}")

    # create emptying stage
    body.append(f"\n{dtab}# empty the pipeline")
    for idx in range(1, n_parts):
        for i, s in enumerate(statements[:n_parts - idx]):
            if use_streams:
                body.append(
                    f"with self.stream({n_parts-i-1},{idx+i-n_parts}):")
                body.append(
                    f"{tab}self.wait_stream({n_parts-i-1},{idx+i-n_parts})")
                body.append(f"{tab}{s}")
            else:
                body.append(s)
        for o in created_after_i[idx]:
            body.append(f"{o}_chunks.append({o})")
        body.append("")

    body.extend(generate_merge_mb(graph, model_outputs))

    pipelined_forward_function = [decleration] + body + [f"return {outputs}"]

    return f"\n{dtab}".join(pipelined_forward_function)


def generate_get_inputs_splits_and_aggeragators(model_inputs: List[str], model_outputs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    # split inputs and and create input generators
    body = [f"# chunk inputs"]
    get_inputs = []
    for i in model_inputs:
        body.extend([f"assert {i}.size(self.batch_dim) >= self.num_chunks",
                     f"{i}_chunks = iter({i}.split({i}.size(self.batch_dim) // self.num_chunks, dim=self.batch_dim))"])
        get_inputs.append(f"{i} = next({i}_chunks)")

    # create chunk aggregators
    body.append(f"\n{dtab}# create output chunk placeholders")
    collect_outputs = []
    for o in model_outputs:
        body.append(f"{o}_chunks = []")
        collect_outputs.append(f"{o}_chunks.append({o})")

    return get_inputs, body, collect_outputs


def generate_merge_mb(graph: Graph, model_outputs: List[str]) -> List[str]:
    # cat chunks
    l = []
    l.append(f"# merge output chunks")
    for n, o in zip(graph.outputs, model_outputs):
        if len(n.tensor_shape) == 1 and len(n.tensor_shape) == 1:
            # scalar
            l.append(f"{o} = sum({o}_chunks)")
        else:
            # tensor
            l.append(f"{o} = torch.cat({o}_chunks,dim=self.batch_dim)")
    l.append("")
    return l


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
