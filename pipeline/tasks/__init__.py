from .interface import DLTask
from .cv_task import CVTask
from .seperate_xy_cv_task import CVTask as CVTaskSepXY
from .seperate_xy_lm_task import LMTask

AVAILABLE_TASKS = {
    'cv': CVTask,
    'cv_sep': CVTaskSepXY,
    # 'lm': LMTask,
    'lm_sep': LMTask
}
