import transformers
from models.normal.NLP_models.modeling_roberta import RobertaForSequenceClassification
from models.partitioned.roberta.roberta_builtin_flatten import create_pipeline_configuration,layerDict,tensorDict
from misc.partition_analysis import run_partitions
from misc.analysis_utils import convert_to_analysis_format

import torch
import sys
import re


def reset_grad(model):
    for p in model.parameters():
        p.grad=None

if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        print("need at least 2 GPUS")
        sys.exit()

    roberta_checkpoint = "roberta-large-mnli"
    modified_device = torch.device("cuda:0")
    original_device = torch.device("cuda:1")
    seq_len = 100
    n_seq = 16

    tokenizer = transformers.AutoTokenizer.from_pretrained(roberta_checkpoint)
    model_config = transformers.AutoConfig.from_pretrained(roberta_checkpoint)


    original_model = transformers.AutoModelForSequenceClassification.from_pretrained(roberta_checkpoint).train().to(original_device)
    modified_model = RobertaForSequenceClassification.from_pretrained(roberta_checkpoint,config=model_config).train()

    print(f"huggingface will run on {original_device}")
    print(f"our model will run on {modified_device}")
    print()

    inputs = tokenizer([" ".join(["a" for _ in range(seq_len)]) for _ in range(n_seq)], return_tensors="pt")
    inputs = (inputs['input_ids'],inputs['attention_mask'])

    config = create_pipeline_configuration(DEBUG=True)

    for i in range(8):
        config['stages'][i]['devices'] = [modified_device]
    
    config = convert_to_analysis_format(config,layerDict(modified_model,basic_blocks=config['basic_blocks']),tensorDict(modified_model))

    assert modified_model.device == modified_device
    assert original_model.device == original_device



    torch.manual_seed(0)

    print("comparing loss between huggingface and the modified model")
    expected_loss = original_model(*[i.to(original_device) for i in inputs])[0].sum()
    torch.cuda.synchronize(original_device)
    actual_loss = modified_model(*[i.to(modified_device) for i in inputs]).sum()
    torch.cuda.synchronize(modified_device)


    if torch.allclose(expected_loss.detach().cpu(),actual_loss.detach().cpu()):
        print("loss is identical")
    else:
        print("loss is different")
    
    print("comparing gradients between huggingface and modified model")
    expected_loss.backward()
    torch.cuda.synchronize(original_device)

    actual_loss.backward()
    torch.cuda.synchronize(modified_device)
    del expected_loss,actual_loss


    expected_gradients = {k: p.grad if p.grad is None else p.grad.detach().cpu() for k,p in original_model.named_parameters()}

    actual_gradients = {re.sub(r'([0-9]+.)',r'layer.\1',k): p.grad if p.grad is None else p.grad.detach().cpu() for k,p in modified_model.named_parameters()}


    assert set(expected_gradients.keys()) == set(actual_gradients.keys()),"state dicts do not contain the same keys"

    n_same = 0
    for k,expected_g in expected_gradients.items():
        actual_g = actual_gradients[k]

        if expected_g is None:
            if actual_g is None:
                n_same+=1
        else:
            if torch.allclose(expected_g,actual_g):
                n_same+=1

    
    print(f"we have {n_same} matching gradient out of {len(actual_gradients)}")



    print()
    reset_grad(modified_model)
    reset_grad(original_model)
    torch.cuda.synchronize(modified_device)
    torch.cuda.synchronize(original_device)




    torch.manual_seed(0)

    print("comparing loss between huggingface and the partitioned model")
    expected_loss = original_model(*[i.to(original_device) for i in inputs])[0].sum()
    torch.cuda.synchronize(original_device)

    #run partitions is hardcoded for default device probably cuda:0
    actual_loss = run_partitions(inputs,config)[0].sum()
    torch.cuda.synchronize(modified_device)


    if torch.allclose(expected_loss.detach().cpu(),actual_loss.detach().cpu()):
        print("loss is identical")
    else:
        print("loss is different")
    
    print("comparing gradients between huggingface and the partitioned model")
    expected_loss.backward()
    torch.cuda.synchronize(original_device)

    actual_loss.backward()
    torch.cuda.synchronize(modified_device)
    del expected_loss,actual_loss


    expected_gradients = {k: p.grad if p.grad is None else p.grad.detach().cpu() for k,p in original_model.named_parameters()}

    actual_gradients = dict()
    scope_to_stage_id = dict()
    for k,p in modified_model.named_parameters():
        new_key = re.sub(r'([0-9]+.)',r'layer.\1',k)
        for i in range(8):
            if k in config[i]['model'].state_dict():
                scope_to_stage_id[new_key] = i
        
        
        actual_gradients[new_key] = p.grad if p.grad is None else p.grad.detach().cpu()



    assert set(expected_gradients.keys()) == set(actual_gradients.keys()),"state dicts do not contain the same keys"

    n_same = 0
    different = []
    for k,expected_g in expected_gradients.items():
        actual_g = actual_gradients[k]

        if expected_g is None:
            if actual_g is None:
                n_same+=1
            else:
                different.append(k)
        else:
            if torch.allclose(expected_g,actual_g,rtol=1e-6):
                n_same+=1
            else:
                assert actual_g is not None
                different.append(k)
    
    print(f"we have {n_same} matching gradient out of {len(actual_gradients)}")
    
    if different:
        max_abs_diff=0
        print("\n different gradients")
        for k in different:
            expected_g = expected_gradients[k]
            actual_g = actual_gradients[k]
            diff = (expected_g-actual_g).abs().max()
            max_abs_diff = max(max_abs_diff,diff)
            print(k,f"on stage {scope_to_stage_id[k]}")
        print(f"\n largest absolute diff between 2 gradients {max_abs_diff}")