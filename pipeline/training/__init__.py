
from .interface import AnyTrainer
from .dummy_trainer import DummyTrainer
from .cvtrainer import CVTrainer

AVAILABLE_TRAINERS = {'dummy': DummyTrainer, 'cv': CVTrainer}