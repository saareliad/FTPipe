
__all__ = ['get_partitioning', 'SUPPORTED_CONFIGS',
           'create_normal_model_instance']


from .cfg_to_model import get_partitioning
from .cfg_to_model import create_normal_model_instance
from .cfg_to_model import MODEL_CONFIGS
SUPPORTED_CONFIGS = MODEL_CONFIGS.keys()
