from typing import Type, Dict, Union

from .interface import LastPartitionTrainer, ScheduledOptimizationStepMultiPartitionTrainer

from .gap_aware_trainer import gap_aware_trainer_factory, GapAwareTrainerMixin
from .grad_norm.global_grad_norm import global_grad_norm_mixin_trainer_factory
from .grad_norm.local_grad_norm import local_grad_norm_mixin_trainer_factory

from .bert_squad_trainer import SquadTrainer
from .cep_trainer import CEPTrainer
from .cv_trainer import CVTrainer, CVTrainerPerStep
from .glue_trainer import GlueTrainer
from .lm_trainer import LMTrainer
from .t5_trainer import T5Trainer

PipelineSupportedTrainerType = Union[ScheduledOptimizationStepMultiPartitionTrainer, GapAwareTrainerMixin]

AVAILABLE_TRAINERS: Dict[str, Type[PipelineSupportedTrainerType]] = dict()


def register_trainer(name, trainer_cls: Type[PipelineSupportedTrainerType]):
    """Registers trainer with mixins."""
    AVAILABLE_TRAINERS[name] = trainer_cls

    AVAILABLE_TRAINERS[name + "_local_grad_norm"] = local_grad_norm_mixin_trainer_factory(trainer_cls=trainer_cls)
    AVAILABLE_TRAINERS[name + "_global_grad_norm"] = global_grad_norm_mixin_trainer_factory(trainer_cls=trainer_cls)
    # NOTE: can mix but "_gap_aware" has to be last
    AVAILABLE_TRAINERS[name + "_gap_aware"] = gap_aware_trainer_factory(trainer_cls=trainer_cls)

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
