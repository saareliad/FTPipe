from typing import Type, Dict, Union

from .cv_trainer import CVTrainer, CVTrainerPerStep
from .gap_aware_trainer import gap_aware_trainer_factory, GapAwareTrainerMixin
from .glue_trainer import GlueTrainer
from .interface import LastPartitionTrainer, LossIncludedInModelMultiPartitionTrainer, \
    DataAndLabelsMultiPartitionTrainer
from .lm_trainer import LMTrainer
from .bert_squad_trainer import SquadTrainer
from .t5_trainer import T5Trainer
from .cep_trainer import CEPTrainer

PipelineSupportedTrainerType = Union[LossIncludedInModelMultiPartitionTrainer, DataAndLabelsMultiPartitionTrainer, GapAwareTrainerMixin]

AVAILABLE_TRAINERS: Dict[str, Type[PipelineSupportedTrainerType]] = dict()


def register_trainer(name, trainer_cls: Type[PipelineSupportedTrainerType]):
    AVAILABLE_TRAINERS[name] = trainer_cls
    ga_trainer_cls = gap_aware_trainer_factory(trainer_cls=trainer_cls)
    AVAILABLE_TRAINERS[name + "_gap_aware"] = ga_trainer_cls


register_trainer("cv", CVTrainer)
register_trainer("cv_per_step_lr_scheduler", CVTrainerPerStep)
register_trainer("lm", LMTrainer)
register_trainer("squad", SquadTrainer)
register_trainer("glue", GlueTrainer)
register_trainer("t5", T5Trainer)
register_trainer("cep", CEPTrainer)  # TODO: this is planed to be outside

def get_trainer_cls(args) -> Type[PipelineSupportedTrainerType]:
    trainer_cls = AVAILABLE_TRAINERS.get(args.trainer['type'])
    assert trainer_cls is not None
    return trainer_cls


