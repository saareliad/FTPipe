from .interface import AnyTrainer
from .cv_trainer import CVTrainer, GapAwareCVTrainer
from .lm_trainer import LMTrainer, GapAwareLMTrainer
from .squad_trainer import SquadTrainer, GapAwareSquadTrainer
from .glue_trainer import GlueTrainer, GapAwareGlueTrainer

AVAILABLE_TRAINERS = {
    'cv': CVTrainer,
    'cv_gap_aware': GapAwareCVTrainer,
    'lm': LMTrainer,
    'lm_gap_aware': GapAwareLMTrainer,
    'squad': SquadTrainer,
    'squad_gap_aware': GapAwareSquadTrainer,
    'glue': GlueTrainer,
    'glue_gap_aware': GapAwareGlueTrainer,
}
