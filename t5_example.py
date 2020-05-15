from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import operator
import math
from pytorch_Gpipe.model_profiling.tracer import trace_module,register_new_explicit_untraced_function,register_new_traced_function
from pytorch_Gpipe import compile_partitioned_model

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("tokenizer created")
    model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()
    print("model created")
    input_ids = tokenizer.encode(
        "Hello, my dog is cute", return_tensors="pt").cuda()  # Batch size 1
    print("tokenized input")
    if False:
        outputs = model(input_ids=input_ids,
                        decoder_input_ids=input_ids, lm_labels=input_ids)
        print("forward pass")
        loss, prediction_scores = outputs[:2]
        print("loss")
        print(loss)
        del outputs,loss,prediction_scores
    else:
        register_new_explicit_untraced_function(operator.is_,operator)
        register_new_explicit_untraced_function(operator.is_not,operator)
        register_new_traced_function(math.log,math)
        register_new_traced_function(torch.einsum,torch)
        
        kwargs={"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids}
        for d in range(6):
            print()
            graph=trace_module(model,args=(),kwargs=kwargs,depth=d)
            print(f"traced t5 depth {d}")
            graph.save_as_pdf(f"t5-small_depth{d}",".")
            compile_partitioned_model(graph,model,0,output_file=f"t5-small_depth{d}")