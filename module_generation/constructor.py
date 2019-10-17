from typing import List, Tuple, Dict
from torch.nn import Module
tab = '    '
dtab = tab + tab


# generate a class decl and __init__ method
def generateConstructor(class_name: str, full_names: List[str], layer_classes: Dict[str, Module]) -> Tuple[str, Dict[str, str]]:
    '''creates the partition constructor and the mapping between layers and field ids
    '''
    class_decl = f"class {class_name}(nn.Module):"
    init_dec = f"{tab}def __init__(self, *layers):"
    super_init = f'{dtab}super({class_name}, self).__init__()'
    layer_names = [f'self.l_{idx}' for idx, _ in enumerate(full_names)]
    assert_statements = generate__init__assertGuards(len(full_names))
    layers_init = generate__init__layersStatements(layer_names, full_names,
                                                   layer_classes)
    scope_to_class_field = dict(zip(full_names, layer_names))
    return '\n'.join([class_decl, init_dec, super_init, assert_statements, layers_init]) + '\n', scope_to_class_field


def generate__init__assertGuards(nlayers: int) -> str:
    ''' generate assert guards ensuring we recieve the necessary amount of layers\n
        in the *layers vararg argument of the constructor
    '''
    assert_statements = f"\n{dtab}# protection against bad initialization\n"
    assert_statements += f"{dtab}assert(len(layers) == {nlayers})\n"
    assert_statements += f"{dtab}assert(all(isinstance(l, nn.Module) for l in layers))\n"
    return assert_statements


def generate__init__layersStatements(layer_names: List[str], full_names: List[str], layer_classes: Dict[str, Module]) -> str:
    ''' generates partition field initialization statements\n
        and comments to describe which scope is allocated to which field
    '''
    statements = [f'{dtab}# initializing partition layers\n']

    for idx, (field, full_name) in enumerate(zip(layer_names, full_names)):
        statements.append(f"# {full_name}\n{dtab}{field} = layers[{idx}]")
        class_name = layer_classes[full_name].__name__
        error_msg = f"f'layers[{idx}] is expected to be of type {class_name} but was of type {{type(layers[{idx}])}}'"
        statements.append(
            f"assert isinstance({field},{class_name}) ,{error_msg}")
    return f'\n{dtab}'.join(statements)
