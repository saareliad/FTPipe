from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration,T5Stack
from transformers import T5Tokenizer

import operator
import math
from pytorch_Gpipe.model_profiling.tracer import trace_module,register_new_explicit_untraced_function,register_new_traced_function

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    print("tokenizer created")
    model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()
    print("model created")
    input_ids = tokenizer.encode(
        "Hello, my dog is cute", return_tensors="pt").cuda()  # Batch size 1
    print("tokenized input")
    if True:
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
        register_new_traced_function(T5Stack.get_extended_attention_mask,T5Stack)
        register_new_traced_function(T5Stack.get_head_mask,T5Stack)
        
        kwargs={"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids}
        graph=trace_module(model,args=(),kwargs=kwargs)
        print("traced t5")
        graph.save_as_pdf("t5-small",".")