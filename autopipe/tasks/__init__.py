"""Partitioning tasks, used from CMD"""
import importlib
import os
from typing import Tuple, Type

from autopipe.cmd_parser import Parser
from .partitioning_task import PartitioningTask

REGISTRY = dict()


def register_task(task_name, parser_cls: Type[Parser], partitioner_cls: Type[PartitioningTask]):
    if not isinstance(task_name, str):
        raise ValueError(f"task name must be a string got {task_name} of type {type(task_name).__name__}")
    elif task_name in REGISTRY:
        raise ValueError(f"task {task_name} is already registered with values {REGISTRY[task_name]}")
    elif not issubclass(parser_cls, Parser):
        raise TypeError(f"registered parser must be a subclass of Parser class got {type(parser_cls).__name__}")
    elif not issubclass(partitioner_cls, PartitioningTask):
        raise TypeError(
            f"registered partitioner must be a subclass of Partitioner class got {type(partitioner_cls).__name__}")

    REGISTRY[task_name] = (parser_cls, partitioner_cls)


def get_parser_and_partitioner(task_name) -> Tuple[Type[Parser], Type[PartitioningTask]]:
    if task_name in REGISTRY:
        return REGISTRY[task_name]
    else:
        raise ValueError(f"unknown task {task_name} available tasks {list(REGISTRY.keys())}")


def import_tasks_from_dir(tasks_dir=os.path.dirname(__file__)):
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

            if task_name == "new_t5":
                import transformers
                if transformers.__version__ < ('4.4.1'):
                    continue
            importlib.import_module('.tasks.' + task_name, package="autopipe")


# in order to automatically register all available tasks
import_tasks_from_dir()
