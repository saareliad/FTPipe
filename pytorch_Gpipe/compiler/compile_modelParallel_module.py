from .partition_forward_method import variableNameGenerator
from typing import List, Tuple, Dict
from collections import deque, defaultdict
import inspect
import torch
from ..utils import flatten,unflatten
tab = '    '
dtab = tab + tab



def create_model_parallel_module(config: Dict) -> str:
    '''create a modelParallel version of the partition config
    '''
    n_stages = len(config['stages'])
    partitions = ", ".join(f"partition{i}" for i in range(n_stages))

    class_decl_and_init = "\n".join([
        f"class ModelParallel(nn.Module):",
        f"{tab}def __init__(self,{partitions}):",
        f"{dtab}super(ModelParallel,self).__init__()",
        f"{dtab}self.batch_dim = {config['batch_dim']}",
        f"{dtab}self.streams = None",
        dtab + f"\n{dtab}".join(f"self.stage{i} = partition{i}"
                                for i in range(n_stages))
    ])
    
    forwards = model_parallel_forward(config)

    create_streams = [f"def create_streams(self,num_chunks):",
    "# create a CUDA stream for every chunk for every device num_chunks X num_devices streams",
    "if (self.streams is None) or len(self.streams[0]) != num_chunks:",
    f"{tab}self.streams = []"]
    create_streams += [f"{tab}self.streams.append([torch.cuda.Stream(self.stage{idx}.device) for _ in range(num_chunks)])" for idx in range(n_stages)]
    create_streams = f"\n{dtab}".join(create_streams)

    current_stream = [f"def current_stream(self,stage_id, chunk_id):",
              "# return the stream for the current device and micro batch",
              "return torch.cuda.stream(self.streams[stage_id][chunk_id])"]
    current_stream = f"\n{dtab}".join(current_stream)

    sync_with_prev_tasks = [f"def sync_with_prev_tasks(self,stage_id, chunk_id):",
                   "stream = torch.cuda.current_stream()",
                   "# wait until the mb was cleared by previous partition",
                   "stream.wait_stream(self.streams[stage_id-1][chunk_id])",
                   "# wait until previous mb was cleared by this partition",
                   "stream.wait_stream(self.streams[stage_id][chunk_id-1])"]
    sync_with_prev_tasks = f"\n{dtab}".join(sync_with_prev_tasks)

    states = f",\n{dtab}{dtab}".join(
        [f"**self.stage{i}.state_dict()" for i in range(n_stages)])

    states = f"{{{states}}}"

    state_dict = f"\n{dtab}".join(
        ["def state_dict(self):", f"return {states}"])

    loads = f"\n{dtab}".join([f"self.stage{i}.load_state_dict(state)" for i in range(n_stages)])
    load_state_dict = f"\n{tab}{tab}".join(
        ["def load_state_dict(self,state):", loads])

    buffer_states = f",\n{dtab}{dtab}{tab} ".join(
        [f"self.stage{i}.named_buffers()" for i in range(n_stages)])
    named_buffers = f"\n{dtab}".join(
        [f"def named_buffers(self):", f"return chain({buffer_states})"])

    parameter_states = f",\n{dtab}{dtab}{tab} ".join(
        [f"self.stage{i}.named_parameters()" for i in range(n_stages)])
    named_parameters = f"\n{dtab}".join(
        [f"def named_parameters(self):", f"return chain({parameter_states})"])

    parameters = f"\n{dtab}".join([
        "def parameters(self):",
        f"return [p for _,p in self.named_parameters()]"
    ])

    buffers = f"\n{dtab}".join(
        ["def buffers(self):", f"return [b for _,b in self.named_buffers()]"])

    chunk = f"\n{tab}".join([s[:-1] for s in inspect.getsourcelines(chunk_inputs)[0]])

    merge = f"\n{tab}".join([s[:-1] for s in inspect.getsourcelines(merge_outputs)[0]])


    return "\n" + f"\n\n{tab}".join([
        class_decl_and_init, create_streams, current_stream, sync_with_prev_tasks,
        chunk,merge,*forwards, state_dict, load_state_dict,
        named_buffers, named_parameters, buffers, parameters
    ]) + "\n\n"


def model_parallel_forward(config: Dict) -> List[str]:
    model_inputs = list(config['model_inputs'].keys())
    model_outputs = list(config['model_outputs'].keys())

    body, activations = forward_statements(config, model_inputs,pipelined=False)
    outputs = ",".join([activations[o] for o in model_outputs])
    forward = simple_forward(model_inputs, body, outputs)

    body, activations = forward_statements(config, model_inputs,pipelined=True)
    out_producers = defaultdict(list)
    for idx, io in config['stages'].items():
        for o in io['outputs']:
            if o in model_outputs:
                out_producers[idx].append(activations[o])

    created_upto_i = defaultdict(list)
    created_after_i = defaultdict(list)
    # created_upto_i[idx] = all outputs which are produced by a stage <= idx
    # created_after_i[idx] = all outputs which are produced by a stage >= idx
    for idx, outs in out_producers.items():
        for i in range(idx + 1):
            created_after_i[i].extend(outs)
        for i in range(idx, len(body)):
            created_upto_i[i].extend(outs)

    pipe_forward = pipelined_forward(config, model_inputs,activations,
                                     body, outputs, created_after_i, created_upto_i, use_streams=False)
    pipe_with_streams = pipelined_forward(config, model_inputs,activations,
                                          body, outputs, created_after_i, created_upto_i, use_streams=True)

    return [forward, pipe_forward, pipe_with_streams]


def pipelined_forward(config: Dict, model_inputs: List[str],activations:Dict[str,str], statements: List[str], outputs: str,
                      created_after_i: Dict[int, List[str]],
                      created_upto_i: Dict[int, List[str]],
                      use_streams: bool) -> str:
    model_outputs = outputs.split(",")
    n_parts = len(statements)
    if use_streams:
        decleration = f"def pipelined_forward_with_streams(self,{', '.join(model_inputs)}, num_chunks={n_parts}):"
    else:
        decleration = f"def pipelined_forward(self,{', '.join(model_inputs)}, num_chunks={n_parts}):"
    body, collect_outputs = generate_chunk_inputs_splits_and_aggeragators(config,model_inputs,
                                                                                    model_outputs)
    

    created,delayed_activations,max_delay = delay(config)

    if delayed_activations:
        body.append(f"\n{dtab}#creating delay buffers")
        for act,tmp in activations.items():
            if act in delayed_activations:
                body.append(f"{tmp}_buff = collections.deque(maxlen={max_delay[act]+1})")
        body.append("")

    body.append("assert num_chunks >= len(list(self.children()))")
    if use_streams:
        body = ["self.create_streams(num_chunks)"] + body

    body.append("chunk_id = 0")
    # create filling stage
    body.append(f"\n{dtab}# filling the pipeline feed chunks until the first chunk enters last stage")
    statements = list(reversed(statements))
    for idx in range(1, n_parts):
        for i, sts in enumerate(statements[n_parts - idx:n_parts]):
            if use_streams:
                body.append(f"with self.current_stream({idx-1-i}, chunk_id - {idx-1-i}):")
                body.append(f"{tab}self.sync_with_prev_tasks({idx-1-i}, chunk_id - {idx-1-i})")
                body.extend([f"{tab}{s}" for s in sts])
            else:
                body.extend(sts)
        for o in created_upto_i[idx - 1]:
            body.append(f"{o}_chunks.append({o})")
        body.append("chunk_id += 1")
        body.append("")

    # create steady stage
    body.append(f"# steady phase pipeline is full every stage processes a chunk in parallel")
    body.append(f"for _ in range(num_chunks - {n_parts-1}):")

    for i, sts in enumerate(statements):
        if use_streams:
            body.append(f"{tab}with self.current_stream({n_parts-i-1}, chunk_id - {n_parts-1-i}):")
            body.append(
                f"{dtab}self.sync_with_prev_tasks({n_parts-i-1},chunk_id - {n_parts-1-i})")
            body.extend([f"{dtab}{s}" for s in sts])
        else:
            body.extend([f"{tab}{s}" for s in sts])
    for o in collect_outputs:
        body.append(f"{tab}{o}")
    body.append(f"{tab}chunk_id += 1")

    # create emptying stage
    body.append(f"\n{dtab}# empty the pipeline until the last chunk leaves the last stage")
    for idx in range(1, n_parts):
        for i, sts in enumerate(statements[:n_parts - idx]):
            if use_streams:
                body.append(
                    f"with self.current_stream({n_parts-i-1}, chunk_id - {n_parts-1-i}):")
                body.append(
                    f"{tab}self.sync_with_prev_tasks({n_parts-i-1}, chunk_id - {n_parts-1-i})")
                body.extend([f"{tab}{s}" for s in sts])
            else:
                body.extend(sts)
        
        for o in created_after_i[idx]:
            body.append(f"{o}_chunks.append({o})")
        
        for a in delayed_activations:
            if created[a] < idx:
                body.append(f"{activations[a]}_buff.appendleft(None)")
        
        body.append("chunk_id += 1")
        body.append("")
    
    if use_streams:
        body.append("torch.cuda.current_stream().wait_stream(self.streams[-1][-1])")

    pipelined_forward_function = [decleration] + body + [return_statement(config,activations)]

    return f"\n{dtab}".join(pipelined_forward_function)


def generate_chunk_inputs_splits_and_aggeragators(config:Dict,model_inputs: List[str], model_outputs: List[str]) -> Tuple[List[str], List[str]]:
    # split inputs and and create input generators
    body = [f"# chunk inputs"]
    for i in model_inputs:
        body.append(f"{i} = self.chunk_inputs({i}, {config['model_inputs'][i]['is_batched']}, num_chunks)")

    # create chunk aggregators
    body.append(f"\n{dtab}# create output chunk placeholders")
    collect_outputs = []
    for o in model_outputs:
        body.append(f"{o}_chunks = []")
        collect_outputs.append(f"{o}_chunks.append({o})")

    return body, collect_outputs


def simple_forward(model_inputs: List[str], body: List[str], outputs: str) -> str:
    decleration = f"def forward(self,{', '.join(model_inputs)}):"
    forward_function = [decleration] + body + [f"return {outputs}"]
    return f"\n{dtab}".join(forward_function)


def forward_statements(config: Dict,
                       model_inputs: List[str],pipelined=False) -> Tuple[List[str], Dict[str, str]]:
    '''generates the forward nethod of the model parallel version of the config
    '''
    if pipelined:
        created,delayed_activations,_ = delay(config)
    else:
        delayed_activations = set()

    n_partitions = len(config['stages'])
    arg_gen = variableNameGenerator()

    activations = dict(zip(model_inputs, model_inputs))
    
    parts = deque(range(n_partitions))
    ios = config['stages']
    body = []
    while parts:
        idx = parts.popleft()

        if all(tensor in activations for tensor in ios[idx]['inputs']):
            if not pipelined:
                inputs = (f"{activations[tensor]}"
                                for tensor in ios[idx]['inputs'])
            else:
                inputs = []
                for tensor in ios[idx]['inputs']:
                    if tensor in model_inputs:
                        inputs.append(activations[tensor]+f"[chunk_id - {idx}]")
                    elif tensor in delayed_activations and (idx-created[tensor]) > 1:
                        inputs.append(activations[tensor]+f"_buff[{idx-created[tensor]-1}]")
                    else:
                        inputs.append(activations[tensor])
            
            inputs = ", ".join(inputs)

            delay_statements=[]
            outputs = []
            for o, t in zip(ios[idx]['outputs'], arg_gen):
                activations[o] = t
                if o in delayed_activations:
                    delay_statements.append(f"{t}_buff.appendleft({t})")
                outputs.append(t)

            n_outputs = len(outputs)
            outputs = ", ".join(outputs)

            delay_statements = [f"{outputs} = self.stage{idx}({inputs})"+ ("[0]" if n_outputs == 1 else "")] + delay_statements
            if pipelined:
                body.append(delay_statements)
            else:
                assert len(delay_statements) == 1
                body.extend(delay_statements)
        else:
            parts.append(idx)

    return body, activations


def return_statement(config:Dict,activations:Dict[str,str])->str:
    sts=[]
    for o,info in config['model_outputs'].items():
        variable_name = activations[o]
        is_batched = info['is_batched']
        sts.append(f"self.merge_outputs({variable_name}_chunks,{is_batched})")

    sts = ", ".join(sts)
    return f"return {sts}"


def chunk_inputs(self,inputs,batched,num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for input,is_batched in zip(flatten(inputs),flatten(batched)):
        if is_batched:
            sizes = torch.full((num_chunks,), input.size(self.batch_dim) // num_chunks, dtype=torch.int32)
            sizes[:input.size(self.batch_dim) % num_chunks] += 1
            xs = [x.contiguous() for x in torch.split(input, sizes.tolist(), dim=self.batch_dim)]
        else:
            xs = [x for _ in range(num_chunks)]

        for i,x in enumerate(xs):
            chunks[i].append(x)
    
    return [unflatten(chunk,batched) for chunk in chunks]


def merge_outputs(self,chunks, batched):
    flattened_is_batched = list(flatten(batched))
    buckets = [[] for _ in flattened_is_batched]

    for chunk in chunks:
        for idx,x in enumerate(flatten(chunk)):
            buckets[idx].append(x)
    
    merged = []
    for xs,is_batched in zip(buckets,flattened_is_batched):
        if is_batched:
            merged.append(torch.cat(xs,dim=self.batch_dim))
        else:
            merged.append(sum(xs))
    
    return unflatten(merged,batched)


def delay(config):
    consumers = defaultdict(list)

    created = dict()
    for idx,stage in config['stages'].items():
        for o in stage['outputs']:
            created[o] = idx
        for i in stage['inputs']:
            if i in config['model_inputs']:
                continue
            consumers[i].append(idx)
    
    delayed_activations = set()
    max_delay = defaultdict(lambda: 0)
    for o,c in consumers.items():
        for i in c:
            if (i - created[o]) > 1:
                delayed_activations.add(o)
                max_delay[o] = max(max_delay[o],i - created[o]-1)

    return created,delayed_activations,max_delay

# config structure
# batch_dim
# depth
# basic_blocks
# model_inputs
    # id
    # shape
    #    dtype
    # is_batched
# model_outputs
    # id
    # shape
    #    dtype
    # is_batched

# stages:
#   id
    # inputs
    #   id
    #    shape
    #    dtype
    #    req_grad
    #    is_batched
    # outputs
    #    id
    #    shape
    #    dtype
    #    is_batched