from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration as ourT5, T5Model as ourBase
from models.normal.NLP_models.modeling_T5_tied_weights import T5ForConditionalGeneration as TiedT5, T5Model as TiedBase
from transformers import T5Tokenizer,T5ForConditionalGeneration as refT5 ,T5Model as refBase
import torch
import operator
import math
from pytorch_Gpipe.model_profiling.tracer import trace_module,register_new_explicit_untraced_function,register_new_traced_function
from pytorch_Gpipe import pipe_model,compile_partitioned_model
from heuristics import node_weight_function,edge_weight_function
from pytorch_Gpipe.utils import layerDict,tensorDict,flatten
import traceback
import os
import importlib
import numpy as np



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


def check_equivalance(ref,our,kws,training=True,exp_prefix=""):
    register_new_explicit_untraced_function(operator.is_,operator)
    register_new_explicit_untraced_function(operator.is_not,operator)
    register_new_traced_function(math.log,math)
    register_new_traced_function(torch.einsum,torch)
    
    ref.train(training)
    our.train(training)
    seed()
    ref_out = list(flatten(ref(**kws)))
    
    seed()
    out = list(flatten(our(**kws)))
    torch.cuda.synchronize()
    assert len(out) == len(ref_out)

    for e,a in zip(ref_out,out):
        assert torch.allclose(a,e),(e,a)
        
    tensors = tensorDict(our)

    
    phase = "training" if training else "evalutation"
    for d in range(6):
        output_file = f"{exp_prefix}_depth{d}_{phase}"
        if os.path.exists(output_file+".py"):
            os.remove(output_file+".py")

        graph = trace_module(our,kwargs=kws,depth=d)
        compile_partitioned_model(graph,our,0,output_file=output_file)

        layers = layerDict(our,depth=d)

        generated=importlib.import_module(output_file).Partition0(layers,tensors)
        seed()
        out = list(flatten(generated(*list(kws.values()))))
        torch.cuda.synchronize()
        assert len(out) == len(ref_out)

        for e,a in zip(ref_out,out):
            assert torch.allclose(a,e),(e,a)
        
        os.remove(output_file+".py")
        print(f"{output_file} equivalent")
    print()


def get_models_for_comparison(base=True,tied=False):
    ref_cls = refBase if base else refT5

    seed()
    transformer_ref = ref_cls.from_pretrained('t5-small').cuda()

    if base and tied:
        our_cls = TiedBase
    elif base and not tied:
        our_cls = ourBase
    elif not base and tied:
        our_cls = TiedT5
    else:
        our_cls = ourT5
    
    seed()
    our = our_cls.from_pretrained('t5-small').cuda()

    if tied:
        our.make_stateless()

    print("models created")

    return transformer_ref,our


if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("tokenizer created")
    
    input_ids = tokenizer.encode(
        "Hello, my dog is cute", return_tensors="pt").cuda()  # Batch size 1
    
    lm_kwargs={"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids,"use_cache":True}
    kwargs = {"input_ids":input_ids,"decoder_input_ids":input_ids,"use_cache":True}
    print("tokenized input")


    for base_transformer in [True,False]:
        for tied_weights in [True,False]:
            if base_transformer:
                inputs = kwargs
            else:
                inputs = lm_kwargs
            
            ref_model,our_model = get_models_for_comparison(base=base_transformer,tied=tied_weights)
            
            prefix = "base_" if base_transformer else "full_"
            prefix+= "tied" if tied_weights else "untied"
            print(f"comparing {prefix}")
            check_equivalance(ref_model,our_model,kwargs,training=True,exp_prefix=prefix)
            check_equivalance(ref_model,our_model,kwargs,training=False,exp_prefix=prefix)
    

    