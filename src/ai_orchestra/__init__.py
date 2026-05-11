"""Tools for conditioned sine-signal reconstruction experiments."""

from .defaults import DEFAULTS, PipelineDefaults
from .signal_sets import SignalSet, generate_signal_set

__version__ = "0.1.0"

__all__ = [
    "DEFAULTS",
    "PipelineDefaults",
    "SignalSet",
    "__version__",
    "generate_signal_set",
]

