""" hack - copied from Pytorch because the API hides it - so modified models will need less parameters"""

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()