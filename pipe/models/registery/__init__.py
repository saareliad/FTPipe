import importlib
import os

from . import model_handler
from .model_handler import AVAILABLE_MODELS, register_model


# from . import cv
# from . import hf
# from . import vit
# from . import cep
# from . import dummy

def _import_handlers_from_dir(tasks_dir=os.path.dirname(__file__),
                              module_name='.models.registery.', package="pipe"):
    """ Automatically import any Python files in the tasks directory
        in order to automatically register all available tasks
    Args:
        tasks_dir: task dir to import from
    """

    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
                not file.startswith('_')
                and not file.startswith('.')
                and (file.endswith('.py') or os.path.isdir(path))
        ):
            task_name = file[:file.find('.py')] if file.endswith('.py') else file

            importlib.import_module(module_name + task_name, package=package)


_import_handlers_from_dir()
