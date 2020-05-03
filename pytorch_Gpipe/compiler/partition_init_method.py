from typing import List, Tuple, Dict, Set
import re
from itertools import chain
from ..model_profiling import Node
tab = '    '
dtab = tab + tab


def generate_init_method(class_name: str, layers: List[Node],
                         is_param_dict: Dict[str, bool], buff_params: Set[Node]) -> Tuple[str, Dict[Node, str]]:
    '''creates the partition constructor and the mapping between layers and field ids
    '''
    class_decl = f"class {class_name}(nn.Module):"
    init_dec = f"{tab}def __init__(self, layers, tensors):"
    super_init = f'{dtab}super({class_name}, self).__init__()'
    layer_names = [f'self.l_{idx}' for idx, _ in enumerate(layers)]
    layers_init = generate__init__layersStatements(layer_names,
                                                   [n.scope for n in layers])
    partition_fields = dict(zip(layers, layer_names))

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

    device_id = re.search(r'\d+$', class_name).group()

    # we initialize it to expected device if DEBUG then the pipeline will set it to cpu device
    device = f"{dtab}self.device = torch.device('cuda:{device_id}')"

    return '\n'.join([class_decl, init_dec, super_init, layers_init, tensor_init, device, lookup]) + '\n', partition_fields


def generate__init__layersStatements(layer_names: List[str], full_names: List[str]) -> str:
    ''' generates partition field initialization statements\n
        and save the layer scopes in the self.scopes field
    '''
    statements = [f'{dtab}# initializing partition layers',
                  "self.scopes=[]"]

    for field, full_name in zip(layer_names, full_names):
        statements.extend([f"{field} = layers['{full_name}']",
                           f"self.scopes.append('{full_name}')"])
    return f'\n{dtab}'.join(statements)


def generate__init__BuffParamStatements(buffers: List[Node], parameters: List[Node]) -> Tuple[str, Dict[Node, str]]:
    ''' generate the init statements to initialize the partitions free floating bufferes and parameters
        free floating means tat those tensors are not part of any layer in this partition
    '''
    tensor_ids = {}
    lines = []
    if buffers:
        lines.append(f"\n{dtab}# initializing partition buffers")
        for idx, b_node in enumerate(buffers):
            lines.append(
                f"self.register_buffer('b_{idx}',tensors['{b_node.scope}'])")
            tensor_ids[b_node] = f'self.b_{idx}'
    if parameters:
        lines.append(f"\n{dtab}# initializing partition parameters")
        for idx, p_node in enumerate(parameters):
            lines.append(
                f"self.register_parameter('p_{idx}', tensors['{p_node.scope}'])")
            tensor_ids[p_node] = f'self.p_{idx}'

    return f'\n{dtab}'.join(lines) + '\n', tensor_ids


def generateLookup(layers_to_id: Dict[Node, str], tensors_to_id: Dict[str, str]) -> str:
    # first generate lookup table
    {'p_0': 'w',
     'l_1': 'module0.sub1.linear'}
    lookup = []
    for layer_node, id in chain(layers_to_id.items(), tensors_to_id.items()):
        # scope: testMod/Linear[linear0] id: l_0
        # we will have 2 keys: l_0.weight l_0.bias
        # we wish to replace l_0 with linear0
        # resulting in keys: linear0.weight linear0.bias
        # for eg scope testMod/Mod0[a]/Sub[b] => a.b
        fields = re.findall("\[[a-zA-Z0-9_]*\]", layer_node.scope)
        fields = map(lambda s: s[1:-1:], fields)
        prefix = '.'.join(fields)
        # remove the self. part of the id
        lookup.append(f"'{id[5:]}': '{prefix}'")
    lookup = f",\n{dtab}{dtab}{dtab}".join(lookup)
    return f"{dtab}self.lookup = {{ {lookup}}}"
