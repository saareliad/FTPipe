from .automatic_prop import AutomaticPipelinePropagator
from .interface import PipelineDataPropagator

AVAILABLE_PROPAGATORS = {
    'auto': AutomaticPipelinePropagator
}
