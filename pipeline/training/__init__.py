
from .interface import AnyTrainer
from .cvtrainer import CVTrainer, GapAwareCVTrainer
from .big_batch import BigBatchManager
AVAILABLE_TRAINERS = {'cv': CVTrainer, 'cv_gap_aware': GapAwareCVTrainer}