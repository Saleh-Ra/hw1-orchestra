"""Tools for conditioned sine-signal reconstruction experiments."""

from .dataloaders import create_data_loaders
from .dataset import (
    FullyConnectedSignalDataset,
    examples_per_signal_set,
    extract_window,
    make_one_hot,
    total_examples,
    train_test_split_indices,
    valid_window_starts,
)
from .defaults import DEFAULTS, PipelineDefaults
from .signal_sets import SignalSet, generate_signal_set, generate_signal_sets

__version__ = "0.1.0"

__all__ = [
    "DEFAULTS",
    "FullyConnectedSignalDataset",
    "PipelineDefaults",
    "SignalSet",
    "__version__",
    "create_data_loaders",
    "examples_per_signal_set",
    "generate_signal_set",
    "generate_signal_sets",
    "extract_window",
    "make_one_hot",
    "total_examples",
    "train_test_split_indices",
    "valid_window_starts",
]

