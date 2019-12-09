
from .interface import AnyTrainer
from .dummy_trainer import DummyTrainer

AVAILABLE_TRAINERS = {'dummy': DummyTrainer}