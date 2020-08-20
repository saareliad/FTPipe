import torch
from typing import List
from pytorch_Gpipe.model_profiling import Node,NodeTypes


tab = "    "
dtab = tab + tab

def pretty_format_obj(obj,dict_prefix=dtab)->str:
    if isinstance(obj, torch.Size):
        # size is inheriting from tuple which is stupid
        return str(obj)
    elif isinstance(obj, (list, tuple, set)):
        elements = [pretty_format_obj(t) for t in obj]
        if len(elements) == 1 and isinstance(obj,tuple):
            #(a,) one element tuple includs a comma
            elements[0]+=","
        elements = ", ".join(elements)
        if isinstance(obj,tuple):
            l,r="(",")"
        elif isinstance(obj,list):
            l,r="[","]"
        else:
            l,r="{","}"
        return l+elements+r 
    elif isinstance(obj, dict):
        items=[]
        for k,v in obj.items():
            if isinstance(k,str):
                k = f"'{k}'"
            else:
                assert isinstance(k,int)
            items.append(f'{k}: {pretty_format_obj(v,dict_prefix+tab)}')
        items[0]=f"\n{dict_prefix}"+items[0]
        return "{" + f",\n{dict_prefix}".join(items) + "}"
    elif obj is type(None):
        return "None"
    elif obj in [torch.Size,torch.device,torch.dtype]:
        return f"torch.{obj.__name__}"
    elif isinstance(obj,type):
        return obj.__name__
    return str(obj)




def sortedPartitionInputs(partition: List[Node]) -> List[Node]:
    '''return a list of all nodes that are input to this partition\n
       sorted by id
    '''
    inputs = set()
    for node in partition:
        
        #NOTE this is for the edge case where we have unused input
        if node.type is NodeTypes.IN:
            inputs.add(node)
        
        inputs.update([
            n for n in node.in_edges
            if (n.stage_id != node.stage_id) or (n.type == NodeTypes.IN)
        ])

    return sorted(inputs, key=lambda n: n.id)


def partitionOutputs(partition: List[Node],
                           model_outputs: List[Node]) -> List[Node]:
    ''' return all nodes that are outputs of the partition\n
    '''

    def isOutput(n):
        part_output = (n.type != NodeTypes.IN) and any(o.stage_id != n.stage_id for o in n.out_edges)
        return part_output or (n in model_outputs)

    return [n for n in partition if isOutput(n)]

