import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.utils.checkpoint import detach_variable


# Pay attention:
#  if you create a tensor on stream1 and use it on stream2, you need to call tensor.record_stream(stream2)
#  in addition to the usual wait_stream syncs (which are still necessary for the familiar cuda 101 reasons)
#  and even if you add the necessary record_stream()s in forward, it might be still broken in backward. 
#  In our case forward seem to be OK without record_stream()s, although if the memory allocation for 
# .to(destination_device) is synced on default stream of destination device and then used by a different stream,
# the same problem may arise

class MlpLayer(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1204, device = 'cuda:0'):
        super().__init__()
        self.device = device
        torch.manual_seed(1234)
        self.mlp  = torch.nn.Linear(input_size, hidden_size).to(device)
        self.relu = torch.nn.ReLU().to(device)

    def forward(self, x):
        return self.relu(self.mlp(x))
        #return self.mlp(x)

class LayerChunk(nn.Module):
    def __init__(self, input_size, layer_size_list = [1024], device = 'cuda:0'):
        super().__init__()
        self.device = device
        layer_size_list = [input_size]+layer_size_list
        num_layers = len(layer_size_list) - 1
        def get_layer(input_size,hiddn_size):
            return MlpLayer(input_size, hiddn_size, device)

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_size_list[i],layer_size_list[i+1]) for i in range(num_layers)])
    
    def forward(self,x):
        #x=x.to(self.device)
        for i, layer in enumerate(self.layers):
        #    print("--------forwarding layer",i, "on device:", self.device)
            x = layer(x)
        return(x)
 
class LastLayerChunk(nn.Module):
    def __init__(self, input_size, layer_size_list = [1024], device = 'cuda:0'):
        super().__init__()
        self.device = device
        layer_size_list = [input_size]+layer_size_list
        num_layers = len(layer_size_list) - 2
        print("DV-debug: LastLayerChunk.init().device:",device)
        def get_layer(input_size,hiddn_size):
            return MlpLayer(
                input_size,
                hiddn_size,
                device)

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_size_list[i],layer_size_list[i+1]) for i in range(num_layers)])
        torch.manual_seed(5678)
        self.mlp_last = torch.nn.Linear(layer_size_list[-2], layer_size_list[-1]).to(device)

    def forward(self,x):
        for i, layer in enumerate(self.layers):
        #    print("--------forwarding layer",i, "on device:", self.device)
            x = layer(x)
        #print("--------forwarding last layer",i+1, "on device:", self.device)
        return(self.mlp_last(x))



class MlpNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_chunks, mp_ngpus,num_microb, method = "regular"):
        super().__init__()
        #self.include_emb_proj = include_emb_proj
        #if include_emb_proj:
        #    self.word_embeddings = torch.nn.Embedding(vocab_size, input_size)
        num_layers = len(hidden_size_list)
        #layer_size_list = [input_size] + hidden_size_list
        layers_per_chunk = math.ceil(float(num_layers)/num_chunks)
        self.num_microb = num_microb
        self.mp_ngpus = mp_ngpus
        print("DV degbug: gpus:", self.mp_ngpus)
        self.num_chunks = num_chunks
        self.method = method
        def get_chunk(input_size,chunk_layers_size_list,device):
            return LayerChunk(
                input_size, chunk_layers_size_list,
                device)
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        chunk_lists = list(chunks(hidden_size_list,layers_per_chunk))
        chunk_inputs = [input_size]+[cl[-1] for cl in chunk_lists[0:-1]]
        self.chunks = torch.nn.ModuleList([])
        for i, chunk_list in enumerate(chunk_lists[0:-1]):
            input_to_chunk = chunk_inputs[i]
            chunk_gpu = torch.device(i % mp_ngpus)
            print("adding chunk {} on device {} with input {}".format(i,chunk_gpu,input_to_chunk))
            print("chunk layers {}".format(chunk_list))
            self.chunks.append(get_chunk(input_to_chunk,chunk_list,chunk_gpu))
        chunk_gpu=torch.device((i+1) % mp_ngpus)
        print("adding last chunk {} on device {} with input {}".format(i+1,chunk_gpu,chunk_inputs[-1]))
        self.chunks.append(LastLayerChunk(chunk_inputs[-1], chunk_lists[-1], chunk_gpu))
    
    def forward_regular(self,x):
     #regular forward
        for i, chunk in enumerate(self.chunks):
            #print('forward chunk', i, "on device", (i % self.mp_ngpus))
            x = chunk(x.to(torch.device(i % self.mp_ngpus)))
        return x
    
    def forward_split_ub_outer(self,x):
        splits = iter(x.split(int(batch_size/self.num_microb), dim=0))
        #print(torch.cat(list(splits))-x)
        ret = []
        for x_next in splits:
            x = x_next
            for i, chunk in enumerate(self.chunks):
                print('forward chunk', i, "on device", (i % self.mp_ngpus))
                x = chunk(x.to(torch.device(i % self.mp_ngpus)))
            ret.append(x)
        return torch.cat(ret)

    def forward_split_ub_outer_streams(self,x):
        last_gpu = (self.num_chunks - 1) % self.mp_ngpus
        splits = iter(x.split(int(batch_size/self.num_microb), dim=0))
        #print(splits)
        ret = []
        for s, x_next in zip(range(self.num_microb),splits):
            x = x_next
            for i, chunk in enumerate(self.chunks):
                print('forward chunk', i, "on device", (i % self.mp_ngpus))
                cur_gpu = i % self.mp_ngpus
                prev_gpu = (i-1) % self.mp_ngpus
                cur_ub = s % self.num_microb
                prev_ub = (s-1) % self.num_microb
                with torch.cuda.stream(multi_gpu_streams[prev_gpu][cur_ub]):
                    x=x.to(torch.device(cur_gpu))
                with torch.cuda.stream(multi_gpu_streams[cur_gpu][cur_ub]):
                    torch.cuda.current_stream().wait_stream(multi_gpu_streams[cur_gpu][prev_ub])
                    torch.cuda.current_stream().wait_stream(multi_gpu_streams[prev_gpu][cur_ub])
                    x = chunk(x)
            ret.append(x)
        torch.cuda.synchronize(last_gpu)
        return torch.cat(ret)


    def forward_split_ub_inner_no_streams(self,x):
        splits = iter(x.split(int(batch_size/self.num_microb), dim=0))
        #print(splits)
        #ret = []
        chunk_outs = []
        chunk = self.chunks[0]
        print("chunk0:",chunk.device)
        for x_next in splits:
            chunk_outs.append(chunk(x_next.to(torch.device(0))))
        s = enumerate(self.chunks)
        next(s)
        for i, chunk in s:
            print('forward chunk', i, "on device", (i % self.mp_ngpus))
            for c,x in enumerate(chunk_outs):
                chunk_outs[c] = chunk(x.to(torch.device(i % self.mp_ngpus)))
        return torch.cat(chunk_outs)#.to(torch.device(i % self.mp_ngpus))

    def forward_split_ub_inner_stream(self,x):             
        last_gpu = (self.num_chunks - 1) % self.mp_ngpus
        splits = iter(x.split(int(batch_size/self.num_microb), dim=0))
        chunk_outs = []
        chunk = self.chunks[0]
        print("chunk0:",chunk.device)
        for s,x_next in zip(range(self.num_microb),splits):
           #print('x_next:',x_next.size())
            prev_ub = (s-1) % self.num_microb
            with torch.cuda.stream(multi_gpu_streams[0][s]):
                torch.cuda.current_stream().wait_stream(multi_gpu_streams[0][prev_ub])
                chunk_outs.append(chunk(x_next))

        enumchunks = enumerate(self.chunks)
        next(enumchunks)
        for i, chunk in enumchunks:
            print('forward chunk', i, "on device", (i % self.mp_ngpus))
            for s,x in enumerate(chunk_outs):
                cur_gpu = i % self.mp_ngpus
                prev_gpu = (i-1) % self.mp_ngpus
                cur_ub = s % self.num_microb
                prev_ub = (s-1) % self.num_microb
                with torch.cuda.stream(multi_gpu_streams[prev_gpu][cur_ub]):
                    x=x.to(torch.device(cur_gpu))
                with torch.cuda.stream(multi_gpu_streams[cur_gpu][cur_ub]):
                    torch.cuda.current_stream().wait_stream(multi_gpu_streams[cur_gpu][prev_ub])
                    torch.cuda.current_stream().wait_stream(multi_gpu_streams[prev_gpu][cur_ub])
                    chunk_outs[s] = chunk(x)
        torch.cuda.synchronize(last_gpu)
        return torch.cat(chunk_outs)


    def forward(self,x):
        if self.method == "pipeline_no_stream":
            print("running pipeline without streams")  
            return self.forward_split_ub_inner_no_streams(x)
        elif self.method == "pipeline_stream":
            print("running pipeline with streams")  
            return self.forward_split_ub_inner_stream(x)
        else:
            print("running regular") 
            return self.forward_regular(x)
 


def train(model, args):
    num_microb = args.num_microbatches
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    torch.manual_seed(7890)
    for j in range(args.num_batches):
        optimizer.zero_grad()
        print("------------------------------------- ")
        print("----------iteration", j, "---------------")
        print("--------------------------------------")
        x = torch.randn(int(args.batch_size), args.input_size, requires_grad=True).to('cuda:0')
        target_var = torch.ones(int(args.batch_size),dtype=torch.long)
        output = model(x)
        target_var = target_var.to(output.device)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()
    # returning stats for comparison:
    return output, loss, model.chunks[-1].layers[0].mlp.weight, model.chunks[-1].mlp_last.weight

if __name__ == "__main__":
    ### import packages ###
    import sys
    import os
    #import io
    #import collections
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Piepline parallel MLP"
    )
    parser.add_argument("--input-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8000)
    parser.add_argument("--num-microbatches", type=int, default=1)
    parser.add_argument("--num-mp-gpus", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--mlp-layers-pattern", type=str, default='2048-2048')
    parser.add_argument("--layer-repeat", type=int, default=1, help="example: repeat('1024-2048',2) = '1024-1024-2048-2048'")
    parser.add_argument("--layer-tile", type=int, default=32, help="example: tile('1024-2048',2) = '1024-2048-1024-2048'")
    parser.add_argument("--num-batches", type=int, default=5)

    args = parser.parse_args()

    batch_size = args.batch_size
    input_size = args.input_size
    num_chunks = args.num_chunks
    num_microb = args.num_microbatches

    device = torch.device("cuda", 0)
    ngpus = torch.cuda.device_count()  # 1
    print("See {} GPU(s)...".format(ngpus))
    mp_ngpus = min(ngpus,args.num_mp_gpus)
    print("Using {} GPU(s) for model parallel ...".format(mp_ngpus))

    layer_list = list(np.tile(np.repeat(np.fromstring(args.mlp_layers_pattern, dtype=int, sep="-"),args.layer_repeat),args.layer_tile))
    print("Number of layers = ", len(layer_list)
        )
    print("Mlp layers:", layer_list)
        
    multi_gpu_streams = []
    for g in range(mp_ngpus):
        cur_gpu_streams = []
        for i in range(num_microb):
            cur_gpu_streams.append(torch.cuda.Stream(device = g))
        multi_gpu_streams.append(cur_gpu_streams)

    # Actual pipleing with streams
    model_pipe_with_stream = MlpNet(input_size, layer_list, num_chunks, mp_ngpus, num_microb, method="pipeline_stream")
    outps, lossps, weight_all_0ps, weight_all_lastps = train(model_pipe_with_stream, args)

    # Actual pipleing without streams for numeric comparison
    model_pipe_no_stream = MlpNet(input_size, layer_list, num_chunks, mp_ngpus, num_microb, method="pipeline_no_stream")
    outp, lossp, weight_all_0p, weight_all_lastp = train(model_pipe_no_stream, args)



    outp  = outp.to("cpu")
    outps = outps.to("cpu")
    weight_0p     =     weight_all_0p.data.to("cuda:0")
    weight_0ps    =    weight_all_0ps.data.to("cuda:0")
    weight_lastp  =  weight_all_lastp.data.to("cuda:0")
    weight_lastps = weight_all_lastps.data.to("cuda:0")

    grad_0p     =     weight_all_0p.grad.to("cuda:0")
    grad_0ps    =    weight_all_0ps.grad.to("cuda:0")
    grad_lastp  =  weight_all_lastp.grad.to("cuda:0")
    grad_lastps = weight_all_lastps.grad.to("cuda:0")

    print("______________Pipelined w/o Streams vs. With Streams ____________")

    print("out equals?", torch.all(outp.eq(outps)))
    print("out max diff:", torch.max(torch.abs(outp - outps)))

    print("weight 0 equals?:", torch.all(weight_0ps.eq(weight_0p)))
    print("weight 0 max diff?", torch.max(torch.abs(weight_0ps - weight_0p)))

    print("weight last equals?:", torch.all(weight_lastps.eq(weight_lastp)))
    print("weight last max diff", torch.max(torch.abs(weight_lastps - weight_lastp)))    
    
    print("weight gradient 0 equals?:", torch.all(grad_0ps.eq(grad_0p)))
    print("weight gradient 0 max diff?", torch.max(torch.abs(grad_0ps - grad_0p)))
    print("weight gradient last equals?:", torch.all(grad_lastps.eq(grad_lastp)))
    print("weight gradient last max diff", torch.max(torch.abs(grad_lastps - grad_lastp)))    

    print(f"loss pipelined {lossp}, loss w streams {lossps}")


