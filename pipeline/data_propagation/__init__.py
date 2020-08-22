from .interface import PipelineDataPropagator
from .automatic_prop import AutomaticPipelinePropagator

AVAILABLE_PROPAGATORS = {
    'auto': AutomaticPipelinePropagator
}
