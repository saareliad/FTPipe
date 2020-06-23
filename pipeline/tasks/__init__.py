from .interface import DLTask
from .cv_task import CVTask
from .separate_xy_cv_task import CVTask as CVTaskSepXY
from .separate_xy_lm_task import LMTask
from .separate_xy_squad_task import SquadTask
from .separate_xy_glue_task import GlueTask

AVAILABLE_TASKS = {
    'cv': CVTask,  # deprecated
    'cv_sep': CVTaskSepXY,
    'lm_sep': LMTask,
    "squad_sep": SquadTask,
    "glue_sep": GlueTask,
}
