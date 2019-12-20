from .communication import P2PCommunicationHandler as CommunicationHandler  # FIXME:

from .communication import CommunicationHandlerBase, get_auto_comm_handler_cls

from .partition_manager import SinglePartitionManager