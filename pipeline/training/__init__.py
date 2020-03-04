from .interface import AnyTrainer
from .cv_trainer import CVTrainer, GapAwareCVTrainer
from .lm_trainer import LMTrainer, GapAwareLMTrainer
AVAILABLE_TRAINERS = {
    'cv': CVTrainer,
    'cv_gap_aware': GapAwareCVTrainer,
    'lm': LMTrainer,
    'lm_gap_aware': GapAwareLMTrainer
}
