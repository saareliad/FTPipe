from .interface import DLTask
from .cv_task import CVTask
from .separate_xy_cv_task import CVTask as CVTaskSepXY
from .separate_xy_lm_task import LMTask

AVAILABLE_TASKS = {
    'cv': CVTask,
    'cv_sep': CVTaskSepXY,
    # 'lm': LMTask,
    'lm_sep': LMTask
}
