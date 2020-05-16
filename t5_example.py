from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import operator
import math
from pytorch_Gpipe.model_profiling.tracer import trace_module,register_new_explicit_untraced_function,register_new_traced_function
from pytorch_Gpipe import pipe_model
from heuristics import node_weight_function,edge_weight_function
from pytorch_Gpipe.utils import layerDict
import traceback

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
        max_d=0
        kwargs={"input_ids":input_ids,"decoder_input_ids":input_ids,"lm_labels":input_ids}
        for d in reversed(range(max_d+1)):
            print("depth ",d)
            output_file=f"t5-small_depth{d}"
            print()
            try:
                graph=pipe_model(model, 0, kwargs=kwargs,
                        nparts=4,
                        depth=d,
                        node_weight_function=node_weight_function(),
                        edge_weight_function=edge_weight_function(
                            12),
                        use_network_profiler=not True,
                        use_graph_profiler=True,
                        recomputation=True,
                        generate_explicit_del=True,
                        save_memory_mode=False,
                        profile_ops=True,
                        output_file=output_file,
                                           )
                graph.save_as_pdf(output_file,".")
            except Exception as e:
                print()
                print(f"failed depth {d}\n")
                traceback.print_exc()
                print()
                print()