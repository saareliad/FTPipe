import sys
from enum import Enum, auto, unique
from typing import List, Optional, Union

from torch import Tensor


@unique
class COMMAND(Enum):
    '''Enum representing the possible commands recognized by the workers
    '''
    TRAIN = auto()
    EVAL = auto()
    FORWARD = auto()
    BACKWARD = auto()
    TERMINATE = auto()


class EmptyException(Exception):
    def __init__(self, msg):
        self.args = msg,
        sys.exit(self)


class Result():
    ''' a wrapper to an asychronous result can be either data or an exception
    attempting to retrieve the data will trigger the exception (if present)
    '''

    def __init__(self, minibatch: int, data: Optional[Tensor] = None, exc_info: Optional[str] = None, metadata=None, DEBUG_CPU_ONLY: bool = False):
        if DEBUG_CPU_ONLY and isinstance(data, Tensor):
            self.data = data.cpu()
        else:
            self.data = data
        self.exc_info = exc_info
        self.minibatch = minibatch
        self.metadata = metadata

    def get(self) -> Tensor:
        if self.exc_info is None:
            return self.data

        raise EmptyException(self.exc_info)

    def hasException(self) -> bool:
        return not (self.exc_info is None)

    def isValid(self) -> bool:
        return not self.hasException()

    def __str__(self) -> str:
        s = f"minibatch:{self.minibatch} "
        if self.isValid():
            if isinstance(self.data, Tensor):
                return s + f"tensor with shape {self.data.shape}"
            else:
                return s + f"{type(self.data).__name__} {self.data} with metadata {self.metadata}"
        else:
            return s + f"exception {self.exc_info}"

    def __repr__(self) -> str:
        return str(self)


Data = Union[Result, List[Result]]
