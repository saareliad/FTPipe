from typing import List, Tuple, Dict, Set
import re
from itertools import chain
from torch.nn import Module
tab = '    '
dtab = tab + tab


def generateConstructor(class_name: str, full_names: List[str], layer_classes: Dict[str, Module],
                        is_param_dict: Dict[str, bool], buff_param_names: Set[str]) -> Tuple[str, Dict[str, str]]:
    '''creates the partition constructor and the mapping between layers and field ids
    '''
    class_decl = f"class {class_name}(nn.Module):"
    init_dec = f"{tab}def __init__(self, layers, buffers, parameters):"
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
    device = f"{dtab}self.device = torch.device('cuda:{device_id}')"

    return '\n'.join([class_decl, init_dec, super_init, layers_init, tensor_init, device, lookup]) + '\n', scope


def generate__init__layersStatements(layer_names: List[str], full_names: List[str], layer_classes: Dict[str, Module]) -> str:
    ''' generates partition field initialization statements\n
        and comments to describe which scope is allocated to which field
    '''
    statements = [f'{dtab}# initializing partition layers',
                  generate__init__assertGuards(len(layer_names))]

    for field, full_name in zip(layer_names, full_names):
        statements.extend([f"# {full_name}",
                           f"assert '{full_name}' in layers, 'layer {full_name} was expected but not given'",
                           f"{field} = layers['{full_name}']"])
        class_name = layer_classes[full_name].__name__
        error_msg = f"f'layers[{full_name}] is expected to be of type {class_name} but was of type {{type({field})}}'"
        statements.append(
            f"assert isinstance({field},{class_name}) ,{error_msg}")
    return f'\n{dtab}'.join(statements)


def generate__init__assertGuards(nlayers: int) -> str:
    ''' generate assert guards ensuring we recieve the necessary amount of layers\n
        in the *layers vararg argument of the constructor
    '''
    assert_statements = f"assert isinstance(layers,dict), f'expected layers to be of type dict but got type{{type(layers)}}'\n"
    assert_statements += f"{dtab}assert(len(layers) == {nlayers})\n"
    assert_statements += f"{dtab}assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'\n"
    assert_statements += f"{dtab}assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'"
    return assert_statements


def generate__init__BuffParamStatements(buffers: List[str], parameters: List[str]) -> str:
    tensor_ids = {}
    lines = [f"\n{dtab}# initializing partition buffers",
             f"assert isinstance(buffers,dict), f'expected buffers to be of type dict got {{type(buffers)}}'",
             f"assert len(buffers) == {len(buffers)}, f'expected buffers to have {len(buffers)} elements but has {{len(buffers)}} elements'",
             f"assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'",
             f"assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'"]
    for idx, b_name in enumerate(buffers):
        lines.extend([f"# {b_name}",
                      f"assert '{b_name}' in buffers, '{b_name} buffer was expected but not given'",
                      f"self.register_buffer('b_{idx}',buffers['{b_name}'])"])
        tensor_ids[b_name] = f'self.b_{idx}'

    lines.extend([f"\n{dtab}# initializing partition parameters",
                  f"assert isinstance(parameters,dict), f'expected parameters to be of type dict got {{type(parameters)}}'",
                  f"assert len(parameters) == {len(parameters)}, f'expected parameters to have {len(parameters)} elements but has {{len(parameters)}} elements'",
                  f"assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'",
                  f"assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'"])
    for idx, p_name in enumerate(parameters):
        lines.extend([f"# {p_name}",
                      f"assert '{p_name}' in parameters, '{p_name} parameter was expected but not given'",
                      f"self.p_{idx} = parameters['{p_name}']"])
        tensor_ids[p_name] = f'self.p_{idx}'

    return f'\n{dtab}'.join(lines), tensor_ids


def generateLookup(layers_to_id, tensors_to_id):
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
