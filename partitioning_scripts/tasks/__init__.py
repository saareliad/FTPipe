from typing import Tuple
import os
import importlib
from .task import Parser,Partitioner


REGISTRY = dict()


def register_task(task_name,parser_cls:Parser,partitioner_cls:Partitioner):
    if not isinstance(task_name,str):
        raise ValueError(f"task name must be a string got {task_name} of type {type(task_name).__name__}")
    elif task_name in REGISTRY:
        raise ValueError(f"task {task_name} is already registered with values {REGISTRY[task_name]}")
    elif not issubclass(parser_cls,Parser):
        raise TypeError(f"registered parser must be a subclass of Parser class got {type(parser_cls).__name__}")
    elif not issubclass(partitioner_cls,Partitioner):
        raise TypeError(f"registered partitioner must be a subclass of Partitioner class got {type(partitioner_cls).__name__}")

    REGISTRY[task_name] = (parser_cls,partitioner_cls)


def get_parser_and_partitioner(task_name)->Tuple[Parser,Partitioner]:
    if task_name in REGISTRY:
        return REGISTRY[task_name]
    else:
        raise ValueError(f"unknow task {task_name} available tasks {list(REGISTRY.keys())}")




# automatically import any Python files in the tasks directory
# in order to automatically register all available tasks
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        task_name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('partitioning_scripts.tasks.' + task_name)