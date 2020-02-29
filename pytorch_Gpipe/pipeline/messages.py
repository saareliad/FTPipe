from enum import Enum, auto, unique


@unique
class COMMAND(Enum):
    '''Enum representing the possible commands recognized by the workers
    '''
    TRAIN = auto()
    EVAL = auto()
    FORWARD = auto()
    BACKWARD = auto()
    LR_STEP = auto()
    TERMINATE = auto()
