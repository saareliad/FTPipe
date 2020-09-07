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
