from typing import List, Tuple, Dict
import re
from itertools import chain
from ..model_profiling import Node
from ..utils import nested_map
from .utils import sortedPartitionInputs,pretty_format_obj
tab = '    '
dtab = tab + tab


def generate_init_method(nodes:List[Node],class_name: str, layers: List[Node],
                         is_param_dict: Dict[str, bool], buff_params: List[Node]) -> Tuple[str, Dict[Node, str]]:
    '''creates the partition constructor and the mapping between layers and field ids
    '''
    
    device_id = re.search(r'\d+$', class_name).group()
    class_decl = f"class {class_name}(nn.Module):"

    layer_scopes_field, tensor_scope_field = generate_layer_and_tensor_scopes(layers,
                                                                              buff_params)

    init_dec = f"{tab}def __init__(self, layers, tensors, device='cuda:{device_id}'):"
    super_init = f'{dtab}super().__init__()'
    layers_init, partition_fields = generate__init__layersStatements(layers)

    params, buffs = [], []
    for n in buff_params:
        if is_param_dict[n.scope]:
            params.append(n)
        else:
            buffs.append(n)

    tensor_init, tensor_ids = generate__init__BuffParamStatements(buffs,
                                                                  params)
    lookup = generateLookup(partition_fields, tensor_ids)
    partition_fields.update(tensor_ids)

    # initialize device and move
    device = f"{dtab}self.device = torch.device(device)"
    move = f"{dtab}self.to(self.device)"

    structure = nested_map(lambda x:1,[n.req_grad for n in sortedPartitionInputs(nodes)])

    cfg = f"{dtab}self.input_structure = {pretty_format_obj(structure)}"

    return '\n'.join([class_decl, layer_scopes_field, tensor_scope_field, init_dec, super_init, layers_init, tensor_init, device, cfg, lookup,move]) + '\n', partition_fields


def generate_layer_and_tensor_scopes(layers: List[Node], buff_params: List[Node]):
    scope_field = ["LAYER_SCOPES=["]
    for n in layers:
        scope_field.append(f"{tab}'{n.scope}',")
    scope_field.append("]")
    scope_field = tab + f"\n{dtab}".join(scope_field)

    tensor_field = ["TENSORS=["]
    for n in buff_params:
        tensor_field.append(f"{tab}'{n.scope}',")
    tensor_field.append("]")
    tensor_field = tab + f"\n{dtab}".join(tensor_field)

    return scope_field, tensor_field


def generate__init__layersStatements(layers: List[Node]) -> Tuple[str, Dict[Node, str]]:
    ''' generates partition field initialization statements\n
        and save the layer scopes in the self.scopes field
    '''
    statements = ["#initialize partition layers",
                  "for idx,layer_scope in enumerate(self.LAYER_SCOPES):",
                  f"{tab}self.add_module(f'l_{{idx}}',layers[layer_scope])"]

    partition_fields = dict(
        zip(layers, [f"self.l_{idx}" for idx, _ in enumerate(layers)]))

    return f'\n{dtab}' + f'\n{dtab}'.join(statements), partition_fields


def generate__init__BuffParamStatements(buffers: List[Node], parameters: List[Node]) -> Tuple[str, Dict[Node, str]]:
    ''' generate the init statements to initialize the partitions free floating bufferes and parameters
        free floating means tat those tensors are not part of any layer in this partition
    '''
    statements = ["#initialize partition tensors",
                  "b=p=0",
                  "for tensor_scope in self.TENSORS:",
                  f"{tab}tensor=tensors[tensor_scope]",
                  f"{tab}if isinstance(tensor,nn.Parameter):",
                  f"{dtab}self.register_parameter(f'p_{{p}}',tensor)",
                  f"{dtab}p+=1",
                  f"{tab}else:",
                  f"{dtab}self.register_buffer(f'b_{{b}}',tensor)",
                  f"{dtab}b+=1"]

    tensor_ids = dict(
        zip(buffers, [f"self.b_{idx}" for idx, _ in enumerate(buffers)]))
    tensor_ids.update(
        dict(zip(parameters, [f"self.p_{idx}" for idx, _ in enumerate(parameters)])))

    return f'\n{dtab}' + f'\n{dtab}'.join(statements) + '\n', tensor_ids


def generateLookup(layers_to_id: Dict[Node, str], tensors_to_id: Dict[Node, str]) -> str:
    # first generate lookup table
    {'p_0': 'w',
     'l_1': 'module0.sub1.linear'}
    lookup = []
    for field_node, id in chain(layers_to_id.items(), tensors_to_id.items()):
        # scope: testMod/Linear[linear0] id: l_0
        # we will have 2 keys: l_0.weight l_0.bias
        # we wish to replace l_0 with linear0
        # resulting in keys: linear0.weight linear0.bias
        # for eg scope testMod/Mod0[a]/Sub[b] => a.b
        fields = re.findall("\[[a-zA-Z0-9_]*\]", field_node.scope)
        fields = map(lambda s: s[1:-1:], fields)
        prefix = '.'.join(fields)
        # remove the self. part of the id
        lookup.append(f"'{id[5:]}': '{prefix}'")
    lookup = f",\n{dtab}{dtab}{dtab}".join(lookup)
    return f"{dtab}self.lookup = {{ {lookup}}}"
