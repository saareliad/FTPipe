
from .interface import AnyTrainer
from .cvtrainer import CVTrainer, GapAwareCVTrainer

AVAILABLE_TRAINERS = {'cv': CVTrainer, 'cv_gap_aware': GapAwareCVTrainer}