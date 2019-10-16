
import torch
from pytorch_Gpipe.model_profiling import NodeTypes, graph_builder
from pytorch_Gpipe import partition_with_profiler
from pytorch_Gpipe.utils import traverse_model
import string
from .forward import forward_function
tab = '    '
dtab = tab + tab


def generate_modules(model: torch.nn.Module, *sample_batch, depth=0):
    graph = partition_with_profiler(model, *sample_batch,
                                    max_depth=depth, nparts=2)

    model_name = type(model).__name__
    graph.save(model_name, '.', show_buffs_params=True, show_weights=True)
    layer_dict = {n: m for m, n, _ in traverse_model(model, depth=depth)}

    parts, part_modules = group_layers_and_nodes_by_partition(layer_dict,
                                                              graph.nodes)

    # import torch torch.nn as nn import torch.nn.functional as F
    torch_import = generate_imports()

    lines = [torch_import]

    # the main code generation loop generating a class decl
    # and forward function
    for idx, (part, layers) in enumerate(zip(parts, part_modules)):
        class_name = f'{model_name}Partition{idx}'
        names = [n.scope for n in part]
        class_decl, scope_to_id = class_init_decl(class_name, names)
        forward = forward_function(part, layers, scope_to_id)
        lines.extend([class_decl, forward])

    return lines


# generate a class decl and __init__ method
def class_init_decl(class_name, full_names):
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
    assert_statements += f"{dtab}assert(all(isinstance(l,nn.Module) for l in layers))\n"
    return assert_statements


def __init__layers_statements(layer_names, full_names):
    statements = [f'{dtab}# initializing partition layers\n']

    for idx, (field, full_name) in enumerate(zip(layer_names, full_names)):
        statements.append(
            f"# {full_name}\n{dtab}{field} = layers[{idx}]")

    return f'\n{dtab}'.join(statements)


def group_layers_and_nodes_by_partition(layers, nodes):
    # groups layers and their respective nodes according to their partition
    # TODO if we have less partitions not all indices will appear
    parts = [[] for i in range(len({n.part for n in nodes}))]
    part_modules = [[] for i in range(len({n.part for n in nodes}))]
    for n in nodes:
        if n.type == NodeTypes.LAYER:
            parts[n.part].append(n)
            part_modules[n.part].append(layers[n.scope])
        elif n.type == NodeTypes.OP:
            scope = n.scope
            if 'aten::' in scope:
                # TODO handle torch functions
                func_name = scope.split('aten::')[1].rstrip(string.digits)
                print(func_name)
                assert hasattr(torch, func_name), 'non torch.FuncName function'
            elif 'prim::' in scope:
                if 'Constant' in scope:
                    print(f'constant {scope}')
                elif 'ListConstruct' in scope:
                    print(f'list building {scope}')
    return parts, part_modules


def generate_imports():
    imports = f'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
    disclaimer = '# this is an auto generated file do not edit unless you know what you are doing\n\n'

    return imports + disclaimer
