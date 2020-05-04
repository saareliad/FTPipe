from typing import List, Tuple, Dict
import re
from itertools import chain
from ..model_profiling import Node
tab = '    '
dtab = tab + tab


def generate_init_method(class_name: str, layers: List[Node],
                         is_param_dict: Dict[str, bool], buff_params: List[Node]) -> Tuple[str, Dict[Node, str]]:
    '''creates the partition constructor and the mapping between layers and field ids
    '''
    class_decl = f"class {class_name}(nn.Module):"

    layer_scopes_field, tensor_scope_field = generate_layer_and_tensor_scopes(layers,
                                                                              buff_params)
    basic_blocks_field = generate_basic_blocks_field(layers)

    init_dec = f"{tab}def __init__(self, layers, tensors):"
    super_init = f'{dtab}super({class_name}, self).__init__()'
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

    device_id = re.search(r'\d+$', class_name).group()

    # we initialize it to expected device if DEBUG then the pipeline will set it to cpu device
    device = f"{dtab}self.device = torch.device('cuda:{device_id}')"

    return '\n'.join([class_decl, basic_blocks_field, layer_scopes_field, tensor_scope_field, init_dec, super_init, layers_init, tensor_init, device, lookup]) + '\n', partition_fields


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


def generate_basic_blocks_field(layers: List[Node]):
    basic_blocks = set()
    for n in layers:
        layer_cls = n.scope.rsplit("/", maxsplit=1)[1]
        layer_cls = layer_cls.rsplit("[", maxsplit=1)[0]
        basic_blocks.add(layer_cls)

    field = ["BASIC_BLOCKS=("]
    for b in basic_blocks:
        field.append(f"{tab}{b},")
    field.append(")")

    return tab + f"\n{dtab}".join(field)


def generate__init__layersStatements(layers: List[Node]) -> Tuple[str, Dict[Node, str]]:
    ''' generates partition field initialization statements\n
        and save the layer scopes in the self.scopes field
    '''
    # statements = [f'{dtab}# initializing partition layers']

    # for field, full_name in zip(layer_names, full_names):
    #     statements.append(f"{field} = layers['{full_name}']")

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
    # tensor_ids = {}
    # statements = []
    # if buffers:
    #     statements.append(f"\n{dtab}# initializing partition buffers")
    #     for idx, b_node in enumerate(buffers):
    #         statements.append(
    #             f"self.register_buffer('b_{idx}',tensors['{b_node.scope}'])")
    #         tensor_ids[b_node] = f'self.b_{idx}'
    # if parameters:
    #     statements.append(f"\n{dtab}# initializing partition parameters")
    #     for idx, p_node in enumerate(parameters):
    #         statements.append(
    #             f"self.register_parameter('p_{idx}', tensors['{p_node.scope}'])")
    #         tensor_ids[p_node] = f'self.p_{idx}'

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
