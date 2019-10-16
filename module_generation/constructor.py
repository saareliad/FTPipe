tab = '    '
dtab = tab + tab


# generate a class decl and __init__ method
def generate_constructor(class_name, full_names):
    class_decl = f"class {class_name}(nn.Module):"
    layer_names = [f'self.l_{idx}' for idx, _ in enumerate(full_names)]
    scope_to_id = dict(zip(full_names, layer_names))
    init_dec = f"{tab}def __init__(self, *layers):"
    super_init = f'{dtab}super({class_name}, self).__init__()'
    assert_statements = __init__assert_guards(len(full_names))
    layers_init = __init__layers_statements(layer_names, full_names)
    return '\n'.join([class_decl, init_dec, super_init, assert_statements, layers_init]) + '\n', scope_to_id


def __init__assert_guards(nlayers):
    assert_statements = f"\n{dtab}# protection against bad initialization\n"
    assert_statements += f"{dtab}assert(len(layers) == {nlayers})\n"
    assert_statements += f"{dtab}assert(all(isinstance(l, nn.Module) for l in layers))\n"
    return assert_statements


def __init__layers_statements(layer_names, full_names):
    statements = [f'{dtab}# initializing partition layers\n']

    for idx, (field, full_name) in enumerate(zip(layer_names, full_names)):
        statements.append(
            f"# {full_name}\n{dtab}{field} = layers[{idx}]")

    return f'\n{dtab}'.join(statements)
