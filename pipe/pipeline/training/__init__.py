from typing import Type

from .cv_trainer import CVTrainer
from .gap_aware_trainer import gap_aware_trainer_factory
from .glue_trainer import GlueTrainer
from .interface import AnyTrainer
from .lm_trainer import LMTrainer
from .bert_squad_trainer import SquadTrainer
from .t5_trainer import T5Trainer
from .cep_trainer import CEPTrainer


AVAILABLE_TRAINERS = {}


def register_trainer(name, trainer_cls: Type[AnyTrainer]):
    AVAILABLE_TRAINERS[name] = trainer_cls
    ga_trainer_cls = gap_aware_trainer_factory(trainer_cls=trainer_cls)
    AVAILABLE_TRAINERS[name + "_gap_aware"] = ga_trainer_cls


register_trainer("cv", CVTrainer)
register_trainer("lm", LMTrainer)
register_trainer("squad", SquadTrainer)
register_trainer("glue", GlueTrainer)
register_trainer("t5", T5Trainer)
register_trainer("cep", CEPTrainer)  # TODO: this is planed to be outside
