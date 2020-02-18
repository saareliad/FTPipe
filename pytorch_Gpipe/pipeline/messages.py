import sys
from enum import Enum, auto, unique
from typing import Optional
import torch
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

    def __init__(self, data: Optional[Tensor] = None, exc_info: Optional[str] = None):
        self.data = data
        self.exc_info = exc_info

    def get(self) -> Optional[Tensor]:
        if self.exc_info is None:
            return self.data

        raise EmptyException(self.exc_info)

    def hasException(self) -> bool:
        return not (self.exc_info is None)

    def __repr__(self) -> str:
        return str(self)

    def to(self, device: torch.device):
        if isinstance(self.data, Tensor):
            data = self.data.to(device=device, non_blocking=True)
            return Result(data=data, exc_info=self.exc_info)

        return self
