
__all__ = ['replace_inplace_for_first_innermost_layer_']


def set_inplace_false_(m):
    """ return True if replaced """
    if hasattr(m, "inplace") and m.inplace:
        m.inplace = False
        return True
    return False


def replace_inplace_for_a_given_layer_(model, layer_name="l_0"):
    """ return True if replaced. """
    if hasattr(model, layer_name):
        return set_inplace_false_(getattr(model, layer_name))
    return False


def replace_inplace_for_first_innermost_layer_(model):
    """
    model: torch.nn.Module.

    return True if replaced
    """
    first_innermost_layer, name = get_innnermost_first_layer_and_name(model, '')
    if not name:
        assert first_innermost_layer is model

    return set_inplace_false_(first_innermost_layer)


def get_innnermost_first_layer_and_name(partition, name=''):
    """ 
    Args:
        partition: a torch.nn.Module
        name: is the name for the partition in the calling context

    Returns:
        the innermost layer and its name

    Example:
        >>> import torch
        >>> m = torch.nn.TransformerDecoderLayer(d_model=10, nhead=2)
        >>> layer, name = get_innnermost_first_layer_and_name(m, 'm')
        >>> print(layer, name)

        Linear(in_features=10, out_features=10, bias=True) out_proj
    """
    list_children = list(partition.named_children())
    if not list_children:
        return partition, name

    name, last_layer = list_children[0]
    del list_children
    return get_innnermost_first_layer_and_name(last_layer, name)


if __name__ == "__main__":
    import torch
    assert (replace_inplace_for_first_innermost_layer_(torch.nn.ReLU(True)))