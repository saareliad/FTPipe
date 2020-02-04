from .interface import DLTask
from .cv_task import CVTask
from .seperate_xy_cv_task import CVTask as CVTaskSepXY

AVAILABLE_TASKS = {'cv': CVTask, 'cv_sep': CVTaskSepXY}
