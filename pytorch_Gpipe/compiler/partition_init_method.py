from typing import List, Tuple, Dict, Set
import re
from itertools import chain
from torch.nn import Module
tab = '    '
dtab = tab + tab


def generate_init_method(class_name: str, full_names: List[str], layer_classes: Dict[str, Module],
                         is_param_dict: Dict[str, bool], buff_param_names: Set[str]) -> Tuple[str, Dict[str, str]]:
    '''creates the partition constructor and the mapping between layers and field ids
    '''
    class_decl = f"class {class_name}(nn.Module):"
    init_dec = f"{tab}def __init__(self, layers, tensors):"
    super_init = f'{dtab}super({class_name}, self).__init__()'
    layer_names = [f'self.l_{idx}' for idx, _ in enumerate(full_names)]
    layers_init = generate__init__layersStatements(layer_names, full_names,
                                                   layer_classes)
    scope = dict(zip(full_names, layer_names))

    params, buffs = [], []
    for k, v in is_param_dict.items():
        if k not in buff_param_names:
            continue
        elif v:
            params.append(k)
        else:
            buffs.append(k)

    tensor_init, tensor_ids = generate__init__BuffParamStatements(buffs,
                                                                  params)
    lookup = generateLookup(scope, tensor_ids)
    scope.update(tensor_ids)

    device_id = re.search(r'\d+$', class_name).group()

    # we initialize it to expected device if DEBUG then the pipeline will set it to cpu device
    device = f"{dtab}self.device = torch.device('cuda:{device_id}')"

    return '\n'.join([class_decl, init_dec, super_init, layers_init, tensor_init, device, lookup]) + '\n', scope


def generate__init__layersStatements(layer_names: List[str], full_names: List[str], layer_classes: Dict[str, Module]) -> str:
    ''' generates partition field initialization statements\n
        and comments to describe which scope is allocated to which field
    '''
    statements = [f'{dtab}# initializing partition layers']

    for field, full_name in zip(layer_names, full_names):
        statements.extend([f"# {full_name}",
                           f"{field} = layers['{full_name}']"])
        class_name = layer_classes[full_name].__name__
        error_msg = f"f'layers[{full_name}] is expected to be of type {class_name} but was of type {{type({field})}}'"
        statements.append(
            f"assert isinstance({field},{class_name}) ,{error_msg}")
    return f'\n{dtab}'.join(statements)


def generate__init__BuffParamStatements(buffers: List[str], parameters: List[str]) -> str:
    ''' generate the init statements to initialize the partitions free floating bufferes and parameters
        free floating means tat those tensors are not part of any layer in this partition
    '''
    tensor_ids = {}
    lines = [f"\n{dtab}# initializing partition buffers"]
    for idx, b_name in enumerate(buffers):
        lines.extend([f"# {b_name}",
                      f"self.register_buffer('b_{idx}',tensors['{b_name}'])"])
        tensor_ids[b_name] = f'self.b_{idx}'

    lines.extend([f"\n{dtab}# initializing partition parameters"])
    for idx, p_name in enumerate(parameters):
        lines.extend([f"# {p_name}",
                      f"self.p_{idx} = tensors['{p_name}']"])
        tensor_ids[p_name] = f'self.p_{idx}'

    return f'\n{dtab}'.join(lines), tensor_ids


def generateLookup(layers_to_id: Dict[str, str], tensors_to_id: Dict[str, str]) -> str:
    # first generate lookup table
    {'p_0': 'w',
     'l_1': 'module0.sub1.linear'}
    lookup = []
    for scope, id in chain(layers_to_id.items(), tensors_to_id.items()):
        # scope: testMod/Linear[linear0] id: l_0
        # we will have 2 keys: l_0.weight l_0.bias
        # we wish to replace l_0 with linear0
        # resulting in keys: linear0.weight linear0.bias
        # for eg scope testMod/Mod0[a]/Sub[b] => a.b
        fields = re.findall("\[[a-zA-Z0-9_]*\]", scope)
        fields = map(lambda s: s[1:-1:], fields)
        prefix = '.'.join(fields)
        # remove the self. part of the id
        lookup.append(f"'{id[5:]}': '{prefix}'")
    lookup = f",\n{dtab}{dtab}{dtab}".join(lookup)
    return f"{dtab}self.lookup = {{ {lookup}}}"
