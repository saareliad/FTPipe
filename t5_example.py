from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration as ourT5, T5Model as ourBase
from models.normal.NLP_models.modeling_T5_tied_weights import T5ForConditionalGeneration as TiedT5, T5Model as TiedBase
from transformers import T5Tokenizer,T5ForConditionalGeneration as refT5 ,T5Model as refBase
import torch
import operator
import math
from pytorch_Gpipe.model_profiling.tracer import trace_module,register_new_explicit_untraced_function,register_new_traced_function
from pytorch_Gpipe import pipe_model,compile_partitioned_model,GraphProfiler,execute_graph,build_graph
from pytorch_Gpipe.utils import layerDict,tensorDict,flatten
import os
import importlib
import numpy as np
from collections import Counter
from heuristics import NodeWeightFunction,EdgeWeightFunction

def seed():
    torch.cuda.synchronize()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    torch.cuda.synchronize()


def nested_print(ts,indent=""):
    if isinstance(ts, (list, tuple)):
        print(f"{indent}tuple of:")
        for t in ts:
            nested_print(t,indent+f"  ")
        print(f"{indent}end tuple")
    else:
        assert isinstance(ts,torch.Tensor)
        print(f"{indent}{ts.shape}")


def get_blocks(model):
    blocks=dict()

    for m in model.modules():
        block = type(m)
        blocks[block.__name__] = block

    return blocks

def count_blocks(model):
    blocks=Counter()

    for m in model.modules():
        block = type(m)
        blocks[block.__name__] +=1

    return blocks
    

MAX_DEPTH=6

def register_functions():
    register_new_explicit_untraced_function(operator.is_,operator)
    register_new_explicit_untraced_function(operator.is_not,operator)
    register_new_traced_function(math.log,math)
    register_new_traced_function(torch.einsum,torch)
    


def ensure_tracing_semantics_and_profiling(ref,our,input_kws,training=True,exp_prefix=""):
    register_functions()
    ref.train(training)
    our.train(training)
    seed()
    ref_out = list(flatten(ref(**input_kws)))
    
    seed()
    out = list(flatten(our(**input_kws)))
    torch.cuda.synchronize()
    assert len(out) == len(ref_out)

    for e,a in zip(ref_out,out):
        assert torch.allclose(a,e),(e,a)
        
    tensors = tensorDict(our)

    basic_blocks = get_blocks(our)
    blocks = [basic_blocks[b] for b in ['T5Attention']]

    
    phase = "training" if training else "evalutation"
    for d in range(MAX_DEPTH):
        output_file = f"{exp_prefix}_depth{d}_{phase}"
        if os.path.exists(output_file+".py"):
            os.remove(output_file+".py")
        
        graph = trace_module(our,kwargs=input_kws,depth=d,basic_blocks=blocks)

        compile_partitioned_model(graph,our,0,output_file=output_file)

        layers = layerDict(our,depth=d,basic_blocks=blocks)

        generated=importlib.import_module(output_file).Partition0(layers,tensors)

       #for some depth configs the use_cache flag will be built in 
        try:
            seed()
            out = list(flatten(generated(*list(input_kws.values()))))
            kws = input_kws
        except TypeError:
            seed()
            kws = dict(input_kws.items())
            kws.pop("use_cache")
            out = list(flatten(generated(*list(kws.values()))))
        
        torch.cuda.synchronize()
        assert len(out) == len(ref_out)

        for e,a in zip(ref_out,out):
            assert torch.allclose(a,e),(e,a)
        
        os.remove(output_file+".py")
        print(f"{output_file} equivalent")

        #ensure all profiling options work
        if training:
            for recomputation in [True,False]:
                for profile_ops in [True,False]:
                    profiler = GraphProfiler(recomputation=recomputation, n_iter=2, profile_ops=profile_ops)
                    execute_graph(our, graph, model_args=(), model_kwargs=kws,
                                pre_hook=profiler.time_forward, post_hook=profiler.time_backward)
                    assert len(profiler.get_weights()) > 0
            print(f"{output_file} can be profiled")

    print()


def get_models_for_comparison(base=True,tied=False):
    ref_cls = refBase if base else refT5

    seed()
    transformer_ref = ref_cls.from_pretrained('t5-small').cuda().train()

    if base and tied:
        our_cls = TiedBase
    elif base and not tied:
        our_cls = ourBase
    elif not base and tied:
        our_cls = TiedT5
    else:
        our_cls = ourT5
    
    seed()
    our = our_cls.from_pretrained('t5-small').cuda().train()

    if tied:
        our.make_stateless()

    print("models created")

    return transformer_ref,our



def compare_models(lm_args,args):
    for base_transformer in [True,False]:
        for tied_weights in [True,False]:
            if base_transformer:
                inputs = args
            else:
                inputs = lm_args
            
            ref_model,our_model = get_models_for_comparison(base=base_transformer,tied=tied_weights)
            
            prefix = "base_" if base_transformer else "full_"
            prefix+= "tied" if tied_weights else "untied"
            print(f"comparing {prefix}")
            ensure_tracing_semantics_and_profiling(ref_model,our_model,inputs,training=True,exp_prefix=prefix)
            ensure_tracing_semantics_and_profiling(ref_model,our_model,inputs,training=False,exp_prefix=prefix)



def display_most_used_nodes(graph,threshold=5):
    nodes = list(graph.nodes)
    nodes = sorted(nodes,key=lambda n: len(n.out_edges),reverse=True)
    print()
    print(f"total graph size: {len(nodes)} nodes")
    for n in nodes:
        if len(n.out_edges) >=threshold: 
            print(n.scope,n.id,len(n.out_edges))
            for o in sorted(n.out_edges,key=lambda u:u.id-n.id):
                if n in o.kwargs:
                    print(f"    kwarg:{o.kwargs[n]} of",o.scope,f"diff:{o.id-n.id} diff_percent:{(o.id-n.id)/len(nodes):.2f}")
                else:
                    print(f"    arg:{o.args.index(n)} of",o.scope,f"diff:{o.id-n.id} diff_percent:{(o.id-n.id)/len(nodes):.2f}")
    print()


COMPARE_MODELS=False

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("tokenizer created")
    
    input_ids = tokenizer.encode(
        "Hello, my dog is cute", return_tensors="pt").cuda()  # Batch (1,6)
    input_ids=input_ids.repeat(32,20).contiguous() #Batch (32,120)
    lm_kwargs={"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids,"use_cache":True}
    kwargs = {"input_ids":input_ids,"decoder_input_ids":input_ids,"use_cache":True}
    print("tokenized input")
    print()

    if COMPARE_MODELS:
        compare_models(lm_kwargs,kwargs)
    else:
        register_functions()
        ref,our = get_models_for_comparison(base=False,tied=False)

        c_ref = count_blocks(ref)
        c_our = count_blocks(our)

        for e,n in c_ref.items():
            print(e,n)
        print()
        for e,n in c_our.items():
            print(e,n)

        del ref

        basic_blocks = get_blocks(our)
        blocks = [basic_blocks[b] for b in ["T5Attention"]]

        pipe_model(our,0,kwargs=lm_kwargs,
        save_memory_mode=False,basic_blocks=blocks,force_no_recomp_scopes=None,node_weight_function=NodeWeightFunction(3),edge_weight_function=EdgeWeightFunction(12,3))
