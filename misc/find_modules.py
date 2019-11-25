def find_modules(module, module_name, module_instance, found):
    """
    Recursively find all instances of a specific module inside a module.

    Arguments:
        module {nn.Module} -- Module to search on
        module_name {str} -- Name of the model to search on in the currect context (used to output access string)
        module_instance {nn.Module} -- Class of the module to search
        found {list} -- List to append results to.

    Result will be [(access_string, model),...] inside 'found'.

    # Adapted from facebook XLM repo

    Examples:

    1. Example of finding inside a class comprehended of MODEL_NAMES:
    ```
    for name in self.MODEL_NAMES:
         find_modules(getattr(self, name),
                      f'self.{name}', HashingMemory, self.memory_list)
    ```

    2. Example finding PKMLayer inside txl:
    ```
    from find_modules import find_modules
    found = []
    find_modules(model, 'model', PKMLayer, found)
    print([t[0] for t in found])
    ```
    """

    if isinstance(module, module_instance):
        found.append((module_name, module))
    else:
        for name, child in module.named_children():
            name = ('%s[%s]' if name.isdigit()
                    else '%s.%s') % (module_name, name)
            find_modules(child, name, module_instance, found)

# Utility to access inner model


def get_inner_model(model, access_str, last_is_attr=True):
    """ returns the inner most model.
        i.e model.v
            were v = access_str.split(".")[-1]

    access_str: can be of the form:
         layers[0].m1.layers[i1].m2.layers[i21][i22].m3 ....

    Example: for access_str= m2.layers[i21][i22].m3.attr
         will return m3
    """

    splits = access_str.split(".")
    if last_is_attr:
        splits = splits[:-1]

    m = model
    for i, v in enumerate(splits):
        # print(v)
        if '[' in v:
            # # Access the list
            # Get the list
            il1 = v.find('[')
            ml = v[:il1]
            m = getattr(m, ml)
            il2 = v.find(']')

            # Continue accessing recursively. ([i1][i2][i3]...)
            v = v[il1:]
            while(v.find('[') != -1):
                il1 = v.find('[')
                il2 = v.find(']')
                iil = int(v[il1 + 1:il2])
                m = m[iil]
                v = v[il2 + 1:]
        else:
            m = getattr(m, v)
        return m
