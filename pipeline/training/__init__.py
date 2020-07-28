from .interface import AnyTrainer
from .cv_trainer import CVTrainer, GapAwareCVTrainer
from .lm_trainer import LMTrainer, GapAwareLMTrainer
from .squad_trainer import SquadTrainer, GapAwareSquadTrainer
from .t5_squad_trainer import SquadTrainer as T5SquadTrainer
from .t5_squad_trainer import GapAwareSquadTrainer as T5GapAwareSquadTrainer
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
    't5_squad': T5SquadTrainer,
    't5_squad_gap_aware': T5GapAwareSquadTrainer,
}
