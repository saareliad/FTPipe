from .interface import DLTask
from .automatic_task import AutomaticPipelineTask

# TODO: the task name is somewhat used when preparing pipeline. This will be changed in the future
AVAILABLE_TASKS = {
    'cv': AutomaticPipelineTask,
    'lm': AutomaticPipelineTask,
    "squad": AutomaticPipelineTask,
    "glue": AutomaticPipelineTask,
    "t5_squad": AutomaticPipelineTask, 
}
