from .interface import CommunicationHandlerBase
# from .common_simple_comm import SimpleCommBase
from .bcast import BCASTCommunicationHandler
from .p2p import P2PCommunicationHandler

# TODO: We want to support hybrid comm
# TODO: Add alon's generic comm.

__all__ = ["get_auto_comm_handler_cls", "CommunicationHandlerBase",
           "BCASTCommunicationHandler", "P2PCommunicationHandler"]

from enum import Enum, auto


class CommPolicy(Enum):
    P2P = auto()
    BCAST = auto()


def to_policy(backend, cpu):
    assert backend in {'nccl', 'gloo', 'mpi'}

    if backend == 'mpi' or cpu:
        return CommPolicy.P2P

    return CommPolicy.BCAST


POLICY_TO_COMM = {
    CommPolicy.P2P: P2PCommunicationHandler,
    CommPolicy.BCAST: BCASTCommunicationHandler,
}


def get_auto_comm_handler_cls(backend, cpu):
    return POLICY_TO_COMM[to_policy(backend, cpu)]
