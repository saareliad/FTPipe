from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration as ourT5, T5Model as ourBase
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


def check_equivalance(ref,our,kws,training=True,prefix=""):
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

    for i,(e,a) in enumerate(zip(ref_out,out)):
        # print(i)
        assert torch.allclose(a,e),(e,a)
        
    tensors = tensorDict(our)

    
    phase = "training" if training else "evalutation"
    for d in range(6):
        output_file = f"{prefix}_depth{d}_{phase}"
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

        for i,(e,a) in enumerate(zip(ref_out,out)):
            # print(i)
            assert torch.allclose(a,e),(e,a)
        
        os.remove(output_file+".py")
        print(f"{output_file} equivalent")
    print()




if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("tokenizer created")
    seed()
    ref_full = refT5.from_pretrained('t5-small').cuda()
    seed()
    our_full = ourT5.from_pretrained('t5-small').cuda()
    seed()
    ref_base = refBase.from_pretrained('t5-small').cuda()
    seed()
    our_base = ourBase.from_pretrained('t5-small').cuda()
    print("models created")
    input_ids = tokenizer.encode(
        "Hello, my dog is cute", return_tensors="pt").cuda()  # Batch size 1
    
    lm_kwargs={"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids,"use_cache":True}
    kwargs = {"input_ids":input_ids,"decoder_input_ids":input_ids,"use_cache":True}
    print("tokenized input")

    print("\nchecking base models")
    check_equivalance(ref_base,our_base,kwargs,training=True,prefix="base")
    check_equivalance(ref_base,our_base,kwargs,training=False,prefix="base")
    print("\nchecking full models")
    check_equivalance(ref_full,our_full,lm_kwargs,training=True,prefix="full")
    check_equivalance(ref_full,our_full,lm_kwargs,training=False,prefix="full")